#Tianfu_MultiviewReflectionMappingShader.py
#Adapted from Prof Oliver Cossairt's MultiviewReflectionMappingShader.py
#Modified by Tianfu Wang
################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision.transforms import Resize

# datastructures
from pytorch3d.structures import Meshes, Textures

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate, Transform3d, euler_angles_to_matrix


from pytorch3d.renderer.blending import (
    BlendParams,
    softmax_rgb_blend,
)

import matplotlib.pyplot as plt 

# rendering components
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, Materials, TexturedSoftPhongShader
)

from pytorch3d.renderer.mesh.texturing import interpolate_face_attributes, interpolate_texture_map

def print_memory(str):
    print('+++++++++++++++++' + str + '+++++++++++++++++')
    mem_alloc = torch.cuda.memory_allocated()
    print('%d Mb allocated'% (mem_alloc/(2**20)))
    max_mem_alloc = torch.cuda.max_memory_allocated()
    print('%d Mb max allowable'% (max_mem_alloc/(2**20)))

class Tianfu_MultiviewReflectionMappingShader(nn.Module):
    """
        Reflection Mapping Shader
        The batch dimension is the number of environment maps to render
    """

    def __init__(self, cameras, shader_maps, Np,mem_debug=False, reflect_debug = False, debug = False, screen_size_u = 2, screen_size_v = 2,  imheight = None, imwidth = None):
        super().__init__()
        self.cameras = cameras
        self.shader_maps = shader_maps
        self.mem_debug = mem_debug
        self.reflect_debug = reflect_debug
        self.screen_size_u = screen_size_u
        self.screen_size_v = screen_size_v
        self.imwidth = imwidth
        self.imheight = imheight
        self.Np = Np
        
        if self.imwidth  == None:
            self.imwidth = Np
        if self.imheight == None:
            self.imheight = Np

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        """"
        Reflection mapping shader resampling an environment map using interpolated coordinates
        """
        
        halfpi = torch.tensor(np.pi/2, device=self.cameras.device)
        
        # get the faces, normals, and textures from the mesh
        faces = meshes.faces_packed()  # (F, 3)
        verts = meshes.verts_packed(); # (V, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        faces_normals = vertex_normals[faces]
        faces_verts = verts[faces]
        # texture_maps = meshes.textures.maps_padded()
        # return texels, faces, verts
        V,_ = verts.shape # the number of vertices
        #Fs,_ = faces.shape # the number of faces
        texture_maps = kwargs.get("batch_pattern_maps");

        # dimensions for resampling
        Nv, H_out, W_out, K = fragments.pix_to_face.shape; 
        Nt, C, H_in, W_in   = texture_maps.shape;  # 3 for RGB
        if meshes.textures is None:
            texels = torch.ones((Nv, H_out, W_out, K, C), device=self.cameras.device)

        else:
            try:
                texels = interpolate_texture_map(fragments, meshes)
            except:
                # print("Mesh Texture not found")
                texels = torch.ones((Nv, H_out, W_out, K, C), device=self.cameras.device)


#         print(Nv, H_out, W_out, K)
#         print(faces_verts)
#         print(fragments.bary_coords)
#         print('#Views: ' + str(Nv))
        
        # lets just take the top K=1 face
        K = 1;
        
        # grab the world to screen transform from input arguments
        screenT = kwargs.get("screenTransform");
#         print(screenT.get_matrix())
        
        ################################################################
        # Note that we only want to rasterize once per view,
        # but we apply Nt texture/reflection mapping operations for each view
        ################################################################

        # pixel_verts: (Nv, H, W, K=1, 3) -> (Nv, K=1, H, W, 3) -> (Nv*K=1, H, W, 3)
        pixel_verts = interpolate_face_attributes(
            fragments.pix_to_face[:,:,:,0:K], fragments.bary_coords[:,:,:,0:K,:], faces_verts)
        pixel_verts = pixel_verts.permute(0, 3, 1, 2, 4).view(Nv * K, H_out, W_out, 3)
        
        # pixel_normals: (Nv, H, W, K=1, 3) -> (Nv, K=1, H, W, 3) -> (Nv*K=1, H, W, 3)
        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face[:,:,:,0:K], fragments.bary_coords[:,:,:,0:K,:], faces_normals)
        pixel_normals = pixel_normals.permute(0, 3, 1, 2, 4).view(Nv * K, H_out, W_out, 3)
#         # vertex colors: 
#         diff_map = self.shader_maps['diff_map'].expand(Nv,-1,-1,-1).reshape(V,3)
#         face_colors = diff_map[faces]
#         diff_texels = interpolate_face_attributes(
#             fragments.pix_to_face[:,:,:,0:K], fragments.bary_coords[:,:,:,0:K,:], face_colors)
#         diff_texels = diff_texels.permute(0, 3, 1, 2, 4).view(Nv * K, H_out, W_out, 3)

        # create a raster mask for the background
        raster_mask = fragments.pix_to_face[:,:,:,0:K] >= 0 # (Nv, H, W, K=1);
        # raster_mask: (Nv, H, W, K=1, 3) -> (Nv, K=1, H, W, 3) -> (Nv*K=1, H, W, 3)
        raster_mask = raster_mask.permute(0, 3, 1, 2).view(Nv * K, H_out, W_out) 

        # we don't need the fragments anymore
        del fragments, meshes, faces_normals, faces_verts
        
        
        if self.mem_debug:
            print_memory('pixel_normals')
        
        ################################################################
        # this is the reflection mapping part
        # camera position, vertices, normals are in world coordinates
        ################################################################
        
        # Calculate the view direction
        camera_position = self.cameras.get_camera_center(); # (Nv, 3)
        # camera_position: (Nv, 3) -> (H, W, Nv, 3) -> (Nv, H, W, 3)
        camera_position = camera_position.expand(H_out,W_out,-1,-1).permute(2,0,1,3); 
        
        view_vec = camera_position - pixel_verts
        view_direction = F.normalize(view_vec, p=2, dim=-1, eps=1e-6)
        del view_vec, camera_position
        
        
        # Renormalize the normals in case they have been interpolated.
        pixel_normals = F.normalize(pixel_normals, p=2, dim=-1, eps=1e-6)
        
        # the angle between the view direction and normal
        cos_angle = torch.sum(pixel_normals * view_direction, dim=-1); #(Nv, H, W)
        # No specular reflection if angle is less than 0.
        # mask = (cos_angle > 0).to(torch.float32)

        Nr = 1; # the number of rays to spawn for monte-carlo integration
        Nr2 = Nr*Nr;

        gamma_map = self.shader_maps['gamma_map'].expand(Nv, H_out, W_out, 3)  
        gamma_map = gamma_map[:,:,:,0]# assume third channel is identical - #(Nv, H, W)
        
        spec_tex_color = torch.zeros((Nt, Nv,H_out,W_out,3), device=self.cameras.device)
        for ray in range(Nr2):

#             # compute the reflection of the view_direction about the normal
#             reflect_direction = -view_direction + 2 * (cos_angle[..., None] * pixel_normals) #(Nv, H, W, 3)
            
            spec_reflect_direction = -view_direction + 2 * (cos_angle[..., None] * pixel_normals) #(Nv, H, W, 3)
#             theta = torch.rand_like(spec_reflect_direction.view(Nv*H_out*W_out,3)) # theta is in [0,pi/2] #(Nv*H*W, 3)
#             theta = torch.asin(torch.sin(theta).pow(gamma_map.view(Nv*H_out*W_out,1)))
#             reflect_rot_mat = euler_angles_to_matrix(theta, 'YXZ') #(Nv*H*W, 3, 3)
#             del theta
            
# #             reflect_direction = reflect_rot_mat.bmm(spec_reflect_direction.view(Nv*H_out*W_out,3,1)).view(Nv, H_out, W_out, 3)
#             reflect_direction = Rotate(reflect_rot_mat,device=self.cameras.device).\
#                     transform_normals(spec_reflect_direction.reshape(Nv*H_out*W_out,1,3)).reshape(Nv, H_out, W_out, 3)
#             reflect_direction = spec_reflect_direction
#             del reflect_rot_mat
            
#             spec_int = F.relu(torch.sum(reflect_direction * spec_reflect_direction, dim=-1)).pow(gamma_map) #(Nv, H, W)

            reflect_direction =spec_reflect_direction 
            del spec_reflect_direction
            
            
            spec_int = torch.ones((Nv,H_out,W_out),device=self.cameras.device);
            
            spec_int = spec_int 
            del cos_angle
            

            if self.mem_debug:
                print_memory('reflect_direction')

            ################################################################
            # intersect with the screen plane
            ################################################################

            pixel_verts = pixel_verts.view(Nv, H_out*W_out, 3)
            reflect_direction = reflect_direction.view(Nv, H_out*W_out, 3)

            # compute vertex positions in screen coordinates
            # pixel_verts[0,0, 0] = -0.0620
            # pixel_verts[0,0, 1] =0.0779
            # pixel_verts[0, 0,2] = -0.0248
            screen_verts = screenT.transform_points(pixel_verts);
            # print('screen_verts')
            # print(screen_verts[0,0])
            screen_verts = screen_verts.view(Nv, H_out, W_out, 3);
    #         del pixel_verts
            # compute reflection direction in screen coordinates
            reflect_screen = screenT.transform_normals(reflect_direction)
            


            reflect_screen = reflect_screen.view(Nv, H_out, W_out, 3);

            #make sure that no reflections are directing away from the screen!
            
            sphere_screen = False;
            if (sphere_screen == False):
                reflect_screen[:,:,:,2] = torch.where(reflect_screen[:,:,:,2]> -1e-10,
                            -1e-10* torch.ones_like(reflect_screen[:,:,:,2]),
                            reflect_screen[:,:,:,2]) 

            if self.reflect_debug:
                y_axis = reflect_screen[0, 511,:, 2].cpu()
            # del reflect_direction
            

            # reflection to screen distance
            ray_length = -screen_verts[:,:,:,2:] / reflect_screen[:,:,:,2:]
            # ray length: (Nv, H_out, W_out, 1) -> (Nv, H_out, W_out, 3) 
            ray_length = ray_length.expand(Nv, H_out, W_out, 3);
            
            # compute ray-plane intersection and assign to uv coordinates
            intersect = ray_length*reflect_screen + screen_verts  
            # print(intersect[:, 400, 600])
            if (sphere_screen == True):
                theta = torch.atan2(reflect_screen[:,:,:,0]/torch.sqrt(reflect_screen[:,:,:,0]  * reflect_screen[:,:,:,0] + reflect_screen[:,:,:,2]  * reflect_screen[:,:,:,2]),reflect_screen[:,:,:,2]/torch.sqrt(reflect_screen[:,:,:,0]  * reflect_screen[:,:,:,0] + reflect_screen[:,:,:,2]  * reflect_screen[:,:,:,2])) 
                phi = -torch.atan2( torch.sqrt(reflect_screen[:,:,:,0]  * reflect_screen[:,:,:,0] + reflect_screen[:,:,:,2]  * reflect_screen[:,:,:,2]), reflect_screen[:,:,:,1])
#                 theta = theta - torch.floor(theta)
                theta = theta + np.pi
                theta = theta/np.pi
                theta = torch.where(theta >1, theta - 2, theta)
#                 theta = theta - torch.floor(theta)
                
                res = torch.cat((theta[:,:,:,None] , -(phi[:,:,:,None]/np.pi * 2 + 1)), 3)
                def_pixel_uvs = res
                del theta, phi

            else:
                def_pixel_uvs = intersect[:,:,:,0:2]

            

            
            
            del ray_length, reflect_screen, intersect
            # expand pixel uvs for each texture 
            # def_pixel_uvs: (Nv, H_out, W_out, 2) -> (Nt, Nv, H_out, W_out, 2)
            def_pixel_uvs = def_pixel_uvs.expand(Nt,-1,-1,-1,-1) # (Nt, Nv, H_out, W_out, 2)
            # def_pixel_uvs: (Nt, Nv, H_out, W_out, 2) -> (Nt*Nv, H_out, W_out, 2)
            def_pixel_uvs = def_pixel_uvs.reshape(Nt*Nv,H_out, W_out, 2) # (Nt*Nv, H_out, W_out, 2)
            if (sphere_screen == False):
                def_pixel_uvs = torch.clone(def_pixel_uvs)
                def_pixel_uvs[:,:,:,0] /= (self.screen_size_u/2)
                def_pixel_uvs[:,:,:,1] /= -(self.screen_size_v/2)
        

            
            if self.reflect_debug:
                x_axis = torch.arange(1024, device = "cpu")
                torch.Tensor.ndim = property(lambda self: len(self.shape))  # Fix it
                plt.plot(x_axis, y_axis )  # Works now
                plt.show()
                del y_axis
  


            if self.mem_debug:
                print_memory('def_pixel_uvs')

            ################################################################
            # envirornment mapping by resampling textures
            ################################################################

            # expand the texture maps for each view
            # texture_maps: (Nt, 3, H_in, W_in) -> (Nv, Nt, 3, H_in, W_in) -> (Nt, Nv, 3, H_in, W_in)
            texture_maps2 = texture_maps.expand(Nv,-1,-1,-1,-1).permute(1,0,2,3,4);
            # texture_maps: (Nt, Nv, 3, H_in, W_out) -> (Nv*Nt, 3, H_in, W_in)
            texture_maps2 = texture_maps2.reshape(Nt*Nv, 3, H_in, W_in);
            
            correspondX, correspondY = torch.meshgrid(torch.linspace(-1, 1, 10), torch.linspace(-1, 1, 10))
            correspond = torch.stack((correspondX, correspondY), axis = 2).to(def_pixel_uvs.device).view(-1, 2)
            # print("correspond")
            # print(correspond.shape)
            # print("def_pixel_uvs")
            # print(def_pixel_uvs.shape)
            def_pixel_uvs_flat = torch.clone(def_pixel_uvs).reshape(Nt*Nv * H_out*  W_out, 2)
            
            
            dist = ((def_pixel_uvs_flat[None,:,:] - correspond[:, None, :]) ** 2).sum(axis = 2)
            dist = dist.reshape(correspond.shape[0], H_out, W_out)
            dist += 1000 * ~raster_mask
            # print(raster_mask.shape)
            
            

            # reflection mapping - the is the sampling part
            # uv-coords are defined as [-1:1,-1,1] -> [0:H_in,0:W_in] in pixel coordinates
            spec_texels = F.grid_sample(texture_maps2, def_pixel_uvs)
            # print("hello")
            # print(spec_texels.max())
            # del def_pixel_uvs
            # del texture_maps2
            
            # texels: (Nv*Nt*K, C, H_out, W_out) -> (Nv*Nt, K, C, H_out, W_out) -> (Nv*Nt, K, H_out, W_out, C)
            spec_texels = spec_texels.view(Nt*Nv, K, C, H_out, W_out).permute(0, 1, 3, 4, 2); 
            # for now lets ignore everything but the first of K faces (e.g., ignoring occluded faces)  
            spec_texels = spec_texels[:, 0, :, :, :].squeeze() # (Nv*Nt, H_out, W_out, C)
            # texels: (Nv*Nt, H_out, W_out, C) -> (Nt, Nv, H_out, W_out, C)
            spec_texels = spec_texels.view(Nt, Nv, H_out, W_out, C)
            spec_tex_color += 1/Nr2*spec_texels * spec_int[None,:,:,:,None]
            del spec_texels
            
            
            if self.mem_debug:
                print_memory('texels')
        
        ################################################################
        # compute MONTE-CARLO diffuse shading
        ################################################################
        if self.shader_maps['diff_map'].sum() > 0:
#         if True:
            Nr = 2; # the number of rays to spawn for monte-carlo integration
            Nr2 = Nr*Nr;
            # create random 2D screen coordinates in range [-1,1]
            # use the same set of screen coords for each view and color
            xs = torch.linspace(-1. + 1/Nr,1.-1/Nr,steps=Nr, device=self.cameras.device)
            X,Y = torch.meshgrid(xs,xs)
            screen_coords_xy = torch.cat((X[:,:,None] * self.screen_size_u/2,Y[:,:,None] * self.screen_size_v/2),-1)#(Nr, Nr, 2)
            sample_coords_xy = torch.cat((X[:,:,None],Y[:,:,None]),-1)
            # lookup the lighting color
            # texture_maps: (Nt, C, H_in, W_in)
            light_intensity = F.grid_sample(texture_maps, sample_coords_xy.expand(Nt,-1,-1,-1)) #(Nt, C=3, Nr, Nr)
            light_intensity = light_intensity.view(Nt,C,Nr2).permute(2,0,1) #(Nr2,Nt,C=3)

            if self.mem_debug:
                print_memory('light_intensity')
            
            # compute 3D screen coordinates
            screen_coords_xy = screen_coords_xy.view(Nr2,2)
            screen_coords = torch.cat((screen_coords_xy, \
                                       torch.zeros_like(screen_coords_xy[:,0:1])), 1) #(Nr2, 3)
        
            diffuse_color = torch.zeros((Nt, Nv,H_out,W_out,3), device=self.cameras.device)
            for ray in range(Nr2):
                
#                 # use same screen coordinates for each texture
#                 screen_coords_xy = torch.rand((1, H_out, W_out ,2), dtype=diffuse_color.dtype, device=self.cameras.device) # x-y screen coordinates in [-1,1]
#                 light_intensity = F.grid_sample(texture_maps, screen_coords_xy.expand(Nt,-1,-1,-1)) #(Nt, C=3, H_out, W_out)
#                 light_intensity = light_intensity.permute(0,2,3,1) #(Nt, H_out, W_out, C=3)
#                 screen_coords = torch.cat((screen_coords_xy, \
#                                            torch.zeros_like(screen_coords_xy[:,:,:,0:1])), 3) #(1, H_out, W_out ,3)


                light_vec = screen_coords[ray,None,None,None,:] - screen_verts; #(Nv, H_out, W_out, 3)

                if self.mem_debug:
                    print_memory('light_vec')

                light_sq_dist = light_vec.pow(2).sum(-1) #(Nv, H_out, W_out)

                if self.mem_debug:
                    print_memory('light_sq_dist')

                light_dir = F.normalize(light_vec, p=2, dim=-1, eps=1e-6) #(Nv, H_out, W_out, 3)

                if self.mem_debug:
                    print_memory('light_dir')

                # the angle between the lighting direction and normal
                # pixel_normals: (Nv*K=1, H_out, W_out, 3)            
                cos_angle = torch.mean(light_dir * pixel_normals, dim=-1); #(Nv, H_out, W_out)

                del light_dir
                # del light_vec
                

                # add the r**2 falloff
                cos_angle = cos_angle / light_sq_dist
                del light_sq_dist

                if self.mem_debug:
                    print_memory('cos_angle')

                # add the light intensity to the shading and sum over the rays
                shading = light_intensity[ray,:,None,None,None,:] * cos_angle[None,:,:,:,None]; #(Nt, Nv, H_out, W_out,C=3)                
    #                 shading = light_intensity[:,None,:,:,:] * cos_angle[None,:,:,:,None]; #(Nt, Nv, H_out, W_out,C=3)
                del cos_angle

                # add the diffuse colormap
                diffuse_color += 1/Nr2 * shading;
                del shading

                if self.mem_debug:
                    print_memory('diffuse_color')

                    
            del screen_coords, screen_coords_xy, light_intensity, screen_verts, sample_coords_xy
            torch.cuda.empty_cache()

            if self.mem_debug:
                    print_memory('done')


        else:
            diffuse_color = torch.tensor(0, device=self.cameras.device);
        
        ################################################################
        # compute the final color
        ################################################################

        # expand diffuse/ambient/specular colors for each texture
        ambient_color = self.shader_maps['amb_map'].expand(Nt, Nv, H_out, W_out, C)
        specular_color = self.shader_maps['spec_map'].expand(Nt, Nv, H_out, W_out, C)
        
        # diffuse texture mapping
        # uv-coords are defined as [-1:1,-1,1] -> [0:H_in,0:W_in] in pixel coordinates
        # texture_maps: (H_in, W_in, 3) -> (Nv, H_in, W_in, 3) -> (Nv, 3, H_in, W_in)
        diff_texture_maps = self.shader_maps['diff_map'].expand(Nv, -1, -1, C).permute(0,3,1,2);
#         diff_texture_maps = self.shader_maps['diff_map'].expand(Nv, H_out, W_out, C).permute(0,3,1,2);
        diff_pixel_uvs = pixel_verts.view(Nv, H_out, W_out, 3)[:,:,:,0:2] #(Nv, H_out, W_out, 2)        
        diff_texels = F.grid_sample(diff_texture_maps, diff_pixel_uvs)
        diff_texels = torch.ones_like(diff_texels)
        del diff_pixel_uvs
        
        # texels: (Nv*K, C, H_out, W_out) -> (Nv, K, C, H_out, W_out) -> (Nv, K, H_out, W_out, C)
        diff_texels = diff_texels.view(Nv, K, C, H_out, W_out).permute(0, 1, 3, 4, 2); 
        diff_texels = diff_texels[:, 0, :, :, :] # (Nv, H_out, W_out, C)
        color =  ambient_color* texels[:, :, :,0, :][None,...] + diffuse_color*diff_texels[None,...] * texels[:, :, :,0, :][None,...]+ specular_color*spec_tex_color 
        #         color = diffuse_color + specular_color*spec_texels + ambient_color
        del spec_tex_color, diff_texels, diffuse_color, specular_color, ambient_color, diff_texture_maps, texels
        
        
        
        ################################################################
        # hard blending
        ################################################################
        bg = .0; 
                
        # apply the mask for background

        depthmaps = pixel_verts.view(Nv, H_out, W_out, 3)
        renders = (raster_mask[None,:,:,:,None]*color) + bg*(~raster_mask[None,:,:,:,None]) #(Nt, Nv, H_out, W_out, C)
        del raster_mask, pixel_verts, color
        half_render_size = self.Np//2
        half_imheight = self.imheight//2
        half_imwidth = self.imwidth//2
#         renders[0,:, :,:,0:2] = res
        renders  = renders[:,:,half_render_size - half_imheight:half_render_size + half_imheight,half_render_size - half_imwidth:half_render_size + half_imwidth,:]
        depthmaps = depthmaps[:,half_render_size - half_imheight:half_render_size + half_imheight,half_render_size - half_imwidth:half_render_size + half_imwidth]
        pixel_normals = pixel_normals[:,half_render_size - half_imheight:half_render_size + half_imheight,half_render_size - half_imwidth:half_render_size + half_imwidth,:]
        
        
        reflect_direction = reflect_direction.reshape((1, half_render_size * 2, half_render_size * 2, 3)) 
        reflect_direction = reflect_direction[:, half_render_size - half_imheight:half_render_size + half_imheight,half_render_size - half_imwidth:half_render_size + half_imwidth,:]
        dist = dist[:,half_render_size - half_imheight:half_render_size + half_imheight,half_render_size - half_imwidth:half_render_size + half_imwidth]
        # renders[:,:, :, :, 2]  = 0.299 * renders[:,:, :, :, 0]  + 0.587 * renders[:,:, :, :, 1] + 0.114 * renders[:,:, :, :, 2]
        # renders[:,:, :, :, 1] = renders[:,:, :, :, 2] 
        # renders[:,:, :, :, 0] = renders[:,:, :, :, 2] 
        def_pixel_uvs = def_pixel_uvs[:,half_render_size - half_imheight:half_render_size + half_imheight,half_render_size - half_imwidth:half_render_size + half_imwidth, :]
        # print("disst")
        # print(dist.shape)
        return pixel_normals, def_pixel_uvs, renders , dist, depthmaps