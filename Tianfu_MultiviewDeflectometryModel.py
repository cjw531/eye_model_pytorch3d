################################################################
#Tianfu_MultiviewDeflectometryModel.py
#Adapted from Prof Oliver Cossairt's MultiviewDeflectometryModel.py
#Modified by Tianfu
################################################################
import os
import io
import torch
import numpy as np
import imageio
import torch.nn as nn
import torch.nn.functional as F
import scipy.spatial as sp
from scipy.ndimage import gaussian_filter
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
from ipywidgets import interactive, interact
# from utils import image_grid, plot_camera_scene
# from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

from PIL import Image

from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)


#mesh i/o
from pytorch3d.utils.ico_sphere import ico_sphere
from pytorch3d.io import load_objs_as_meshes, load_obj

# datastructures
from pytorch3d.structures import Meshes, Textures

# 3D transformations functions
from pytorch3d.transforms import euler_angles_to_matrix,Rotate, Translate, Transform3d, so3_log_map, so3_exponential_map,matrix_to_euler_angles
from pytorch3d.renderer.cameras import SfMPerspectiveCameras

# rendering components
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, Materials, TexturedSoftPhongShader
)

# from Tianfu_MultiviewNormalShader import Tianfu_MultiviewNormalShader
from Tianfu_MultiviewReflectionMappingShader import Tianfu_MultiviewReflectionMappingShader 

def gaussian_blur_torch(in_tensor, sigma=0):
    if sigma > 0:
        # create Gaussian matrix
        kernel_width = 4*int(np.ceil(sigma))+1
        xp = np.linspace(-kernel_width/2,kernel_width/2,kernel_width) # grid is scaled -1,1
        Y,X = np.meshgrid(xp, xp);
        G = np.exp(-(X**2 + Y**2)/2/sigma)[None, None, ...]
        G = torch.tensor(G, dtype=in_tensor.dtype).to(in_tensor.device)
        G = G/G.sum()
        
        # create Gaussian weights for each color channel
        if in_tensor.dim() < 3:
            in_tensor = in_tensor[...,None]
        H,W,C = in_tensor.shape            
        w = torch.zeros((C,C,kernel_width,kernel_width), dtype=in_tensor.dtype).to(in_tensor.device)
        for c in range(C):
            w[c,c,:,:] = G;

        # blur the input tensor
        blurred = F.conv2d(in_tensor.permute(2,0,1)[None,...], \
                           weight=w, padding=kernel_width//2).permute(0,2,3,1).squeeze()
        
        return blurred
    else:
        return in_tensor
    

class Tianfu_MultiviewDeflectometryModel(nn.Module):
    def __init__(self, device, Nv=41, Np=128, max_height=.02,
                 dtype=torch.float32, init_depth=None, init_cam=None, init_verts2d=None,
                 shader_maps=None, cameraToScreenTransform=None, screenT=None, screenR=None, screenS=None,
                 depth_est = False, opt_settings = dict({'opt_depth': False, 'opt_norm': False, 'opt_cam': False, 'opt_screen': False,
                 'opt_diff':False, 'opt_spec':False, 'opt_gamma':False, 'opt_amb':False, 'opt_verts2d':False, 'opt_mesh_transform': False,
                     'opt_pattern': False, 'opt_geom': False
                 }),mesh = None, mesh_rotate = None, face_mesh = None, mesh_translate = None, mesh_centofrot = None, screen_size_u = 2, screen_size_v = 2, imheight = None, imwidth = None, logger = None, opt_translate = False, opt_rotate = False,opt_centofrot = False, xy = None):

        ################################################################
        # Nv - the number of vertices on a side for the meshgrid
        # Np - the number of pixels to render each image
        # imheight - the number of height pixels wanted in the output image
        # imwidth - the number of width pixels wanted in output image
        # mesh - the mesh that is rendered in the image
        # mesh_rotate, mesh_translate - additional translation and rotation of the mesh
        # max_height - the height of the depthmap with with width [-1,1]
        # dtype - the CUDA datatype (default Single)
        # init_depth - the depthmap of vertices (default NxN tensor of zeros)
        # init_cam - the initial camera params (3x1 tensor for dist/elev/azim)
        # opt_depth/norm/cam - should we optimize these parameters?
        # screen_size_u, screen_size_v- horizontal and vertical size of the screen 
        
        ################################################################
        # creates the following data memebers used for rendering
        #
        # self.verts2D -  # (1, N*N, 2) tensor of 2D coordinates for meshgrid
        # self.verts_uv - # (1, N*N, 2) tensor of 2D UV coords for vertices
        # self.faces -    # (1, F, 3) tensor of indices for F triangles
        # self.faces_uvs -# (1, F, 3) tensor of uv indices for F triangles
        ################################################################
        super().__init__()

        ################################################################
        # initialize member variables
        ################################################################

        # keep track of the device and max surface height
        self.dtype=dtype;
        self.device=device;
        self.max_height = max_height;
        self.Nv = Nv; # number of vertices in mesh is Nv**2
        # start with empty normals
        self.normals = torch.zeros((Np,Np,3), dtype=dtype)
        self.opt_settings = opt_settings;
        self.mesh = mesh
        self.screen_size_u = screen_size_u
        self.screen_size_v = screen_size_v
        self.mesh_rotate = mesh_rotate
        self.mesh_translate = mesh_translate
        self.Np = Np
        self.imheight = imheight
        self.imwidth = imwidth
        self.logger = logger
        self.face_mesh = face_mesh
        self.mesh_centofrot = mesh_centofrot
        if self.mesh_rotate is None:
            self.mesh_rotate = Rotate(torch.eye(3,3,device=device, dtype=dtype), device = device)
        if self.mesh_translate is None:
            self.mesh_translate = torch.tensor([0,0,0],device=device, dtype=dtype)
        if mesh_centofrot is None:
            self.mesh_centofrot = torch.tensor([0,0,0],device=device, dtype=dtype)
            
        self.smoothing = None
        
        self.meshLogR = so3_log_map(self.mesh_rotate.get_matrix()[:,:3, :3])
        self.meshLogR[torch.isnan(self.meshLogR)] = 0; # set nans to zero
        self.meshLogR = self.meshLogR.to(device)
        self.meshLogR[torch.isnan(self.meshLogR)] = 0; # set nans to zero
        ################################################################
        # initialize shader maps 
        ################################################################

        if shader_maps == None:
            diff_map = torch.zeros((1,1,3),device=device)
            spec_map = torch.ones((1,1,3),device=device)
            gamma_map = 1000*torch.ones((1,1,3),device=device)
            amb_map = torch.zeros((1,1,3),device=device)
            fixed_depthmap = torch.zeros((Nv,Nv,2),device=device) # nothing is fixed
            shader_maps=dict({'diff_map':diff_map, 'spec_map':spec_map,'gamma_map':gamma_map, 'amb_map':amb_map, 'fixed_depthmap':fixed_depthmap})        
        
        self.diff_map = shader_maps['diff_map']
        self.spec_map = shader_maps['spec_map']
        self.gamma_map = shader_maps['gamma_map']
        self.amb_map = shader_maps['amb_map']
        if self.mesh == None:
            self.fixed_depthmap = shader_maps['fixed_depthmap']
        self.shader_maps = shader_maps;
        
        ################################################################
        # initialize mesh 
        ################################################################

        # if there is no depthmap provided, initialize to zeros
        if self.mesh == None:
            if init_depth == None:
                self.Nv = 2
                depthmap = torch.zeros((Nv,Nv, 1), dtype=dtype);
            else:
                self.Nv = init_depth.shape[0];
            self.depthmap = init_depth;
            
            if init_verts2d == None:
                # create a regular grid of spatial coords for uniform mesh
                xp = np.linspace(-1,1,Nv) # grid is scaled -1,1
                Y,X = np.meshgrid(xp, xp);
    
                # create vertices and faces
                init_verts2d = torch.cat((torch.tensor(Y, dtype=dtype).view(Nv*Nv,1), 
                                     torch.tensor(X, dtype=dtype).view(Nv*Nv,1)), 1).to(device) # (1, Nv*Nv, 2)
            self.verts2D = init_verts2d
    
            # use scipy to help with Delaunay triangulation of the faces
            tri = sp.Delaunay(self.verts2D.cpu().numpy());
            self.faces = torch.tensor(tri.simplices, dtype=torch.int64).to(device); # (1, F, 3) - datatype must be integer for indices
        else:
            _, self.faces  = self.mesh.get_mesh_verts_faces(0)

        ################################################################
        # choose initial camera model 
        ################################################################
        
        if init_cam == None:
            self.views = 1;
            # Get the position of the camera based on the spherical angles
            R, T = look_at_view_transform(dist=2, elev=0, azim=0, device=device) # (views,3,3), (views,3) 
            init_cam = SfMPerspectiveCameras(device=device, R=R, T=T);
        
        self.cameras = init_cam;
        
        # optimize the rotation in axis-angle format
        self.camT = self.cameras.T
        self.camLogR = so3_log_map(self.cameras.R); #(views, 3) - want to optimize axis-angle
        self.camLogR[torch.isnan(self.camLogR)] = 0; # set nans to zero

        ################################################################
        # create the screen transforms 
        ################################################################
        if cameraToScreenTransform == None:
            if screenT==None:
                screenT = self.cameras.T;
            if screenR==None:
                screenR = self.cameras.R;
            if screenS==None:
                screenScale = torch.tensor(1);

            # convert to axix-angle format 
            screenLogR = so3_log_map(screenR); #(views, 3) - want to optimize axis-angle
            screenLogR[torch.isnan(screenLogR)] = 0; # set nans to zero

            # create the tranform object
            cameraToScreenTransform =\
                Transform3d(device=device).scale(screenScale)\
                .compose(Rotate(so3_exponential_map(screenLogR)))\
                .translate(screenT).to(device)
        
                
        ################################################################
        # decide what to optimize over
        ################################################################
        self.mesh_scale = torch.tensor([1,1,1],device=device, dtype=dtype)

        if self.opt_settings['opt_mesh_transform']:
            if opt_translate:
#                 self.meshLogR = nn.Parameter(self.meshLogR)    
                # self.mesh_scale = nn.Parameter(self.mesh_scale)
                self.mesh_translate  = nn.Parameter(self.mesh_translate)
            if opt_rotate:
                self.meshLogR = nn.Parameter(self.meshLogR)
            if opt_centofrot:
                self.mesh_centofrot =  nn.Parameter(self.mesh_centofrot)
            # self.elev = nn.Parameter(self.elev)
            # self.azim = nn.Parameter(self.azim)
        if self.opt_settings['opt_depth']:
            self.depthmap = nn.Parameter(self.depthmap)
        if self.opt_settings['opt_verts2d']:
            self.verts2D = nn.Parameter(self.verts2D)            
        if self.opt_settings['opt_norm']:
            self.normals = nn.Parameter(self.normals)
        if self.opt_settings['opt_cam']:
            # optimize the camera positions
            self.camT = nn.Parameter(self.camT)
            self.camLogR = nn.Parameter(self.camLogR)
        if self.opt_settings['opt_screen']:
            # optimize the screen positions
            self.screenT = nn.Parameter(self.screenT)
            self.screenLogR = nn.Parameter(self.screenLogR)
        if self.opt_settings['opt_diff']:
            self.diff_map = nn.Parameter(self.diff_map)
        if self.opt_settings['opt_diff']:
            self.diff_map = nn.Parameter(self.diff_map)
            print("optimizing diff_map")
        if self.opt_settings['opt_amb']:
            self.amb_map = nn.Parameter(self.amb_map)
        if self.opt_settings['opt_spec']:
            self.spec_map = nn.Parameter(self.spec_map)
        if self.opt_settings['opt_gamma']:
            self.gamma_map = nn.Parameter(self.gamma_map)
        if self.opt_settings['opt_pattern']:
            self.pattern_size = 256
            self.pattern = torch.rand(1, 3, self.pattern_size,self.pattern_size, dtype = dtype, device = device)
            self.pattern = nn.Parameter(self.pattern)
        if self.opt_settings['opt_geom']:
            # self.deform = 0.002 * self.mesh.verts_list()[0].clone() + 1e-7
            # self.deform = nn.Parameter(self.deform)
            latitudes = 100
            longitudes  = 100
            self.latitudes = latitudes;
            self.longitudes  = longitudes;
            delta_latitude = np.pi /latitudes;
            delta_logitude = 2 * np.pi/longitudes;
            verts = torch.zeros(((latitudes + 1) * (longitudes + 1), 3), dtype= dtype,device = device)


            latitudeAngles = torch.linspace(np.pi/2, -np.pi/2, latitudes + 1, dtype= dtype,device = device)
            # z = 1.16 * torch.sin(latitudeAngles)[50:]
            # xy = torch.sqrt(torch.max(0.85 ** 2 - (z + 0.31) **2, 1 - z* z  ) )
            z = 1.05 * torch.sin(latitudeAngles)[latitudes//2:]
            if xy == None:
                xy = torch.sqrt(torch.max(0.75 ** 2 - (z + 0.3) **2, 1 - z* z  ) )
            
            self.z_scale = torch.tensor([1.06], dtype = dtype, device = device)
            # self.z_scale = nn.Parameter(self.z_scale)
            self.xy = xy
            self.z = z                
            self.xy = nn.Parameter(self.xy)
        # self.mesh_verts = GenerateSphereSmoothVerts(self.xy)
            self.mesh_faces = GenerateSphereSmoothFaces(latitudes = self.latitudes, longitudes = self.longitudes)
                    
        # the rasterization settings
        self.raster_settings = RasterizationSettings(
            image_size=Np, 
            # blur_radius=np.log(1. / 1e-6 - 1.)*1e-6, 
            blur_radius=0,     
            faces_per_pixel=1, 
            bin_size=0
        )
        
        self.face_mask = None

        ################################################################
        # create the renderer with a ReflectionMappingShader
        ################################################################        
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=self.raster_settings
            ),
            shader=Tianfu_MultiviewReflectionMappingShader(cameras=self.cameras, shader_maps=self.shader_maps, Np = self.Np, screen_size_u = self.screen_size_u, screen_size_v = self.screen_size_v, imheight = self.imheight, imwidth = self.imwidth)
        )
        
        ################################################################
        # expand verts/faces to have same batchdim as textures
        ################################################################
        self.views = self.cameras.R.shape[0];
        self.faces = self.faces.expand(self.views,-1,-1); # (views, F, 3) 
        
        # store screen transform and associated objects to pass to shader        
        # might want to optimize independently
        # self.screenT = screenT.to(device)
        # self.screenLogR = screenLogR.to(device); #(views, 3) - want to optimize axis-angle
        # self.screenScale = screenScale.to(device)

        # self.worldToScreenTransform = self.cameras.get_world_to_view_transform().compose(cameraToScreenTransform)
        a = self.cameras.get_world_to_view_transform().compose(cameraToScreenTransform).get_matrix()[0].expand(2, -1, -1)
        print("cam_to_Screen")
        print(cameraToScreenTransform.get_matrix())
        print("world_to_cam")
        print(self.cameras.get_world_to_view_transform().get_matrix())
        self.worldToScreenTransform = Transform3d(device = device)
        self.worldToScreenTransform._matrix = cameraToScreenTransform.get_matrix()[0].expand(2, -1, -1)
        # self.worldToScreenTransform = cameraToScreenTransform
        # print(self.cameras.get_world_to_view_transform().get_matrix())
        print("world_to_screen")
        print(self.worldToScreenTransform.get_matrix())


    ################################################################
    # Rendering function 
    ################################################################
    def forward(self, view_batchids=[0], texture_batchids=[0], sigma_diff=0, sigma_depth=0):
        if self.mesh == None:
            # clip the depthmap to a reasonable range
            clamped_depth = self.depthmap.clamp(min=-self.max_height, max=self.max_height).to(self.device)
            
            # apply the fixed depthmap values
            mask = self.fixed_depthmap[:,:,0:1];
            fixed_depthvals = self.fixed_depthmap[:,:,1:]
            clamped_depth = mask*fixed_depthvals + (~mask)*clamped_depth
            
            # clamped_depth: (N,N) -> (1,N*N,1)
            clamped_depth = clamped_depth.view(self.Nv*self.Nv,1)
    
            diff_map = self.diff_map
            if sigma_diff>0:
                diff_map = gaussian_blur_torch(diff_map, sigma_diff).clamp(min=0, max=20)
            if sigma_depth>0:
                clamped_depth = clamped_depth.view(self.Nv,self.Nv)
                clamped_depth = gaussian_blur_torch(clamped_depth, sigma_depth)
                clamped_depth = clamped_depth.view(self.Nv*self.Nv,1)
        
        # # create the mesh from the 2D vertices and the depthmap
        # verts3D = torch.cat((self.verts2D, clamped_depth), 1).expand(self.views,-1,-1); # (views,N*N,3)
        # create a batch of textures from the sinusoids.
        batch_texture_maps = self.pattern[texture_batchids,...]
        ################################################################
        # load the 3D mesh
        ################################################################
        if self.mesh != None:
            #create the meshes and apply rotation and translation transformations
            verts3D, self.faces  = self.mesh.get_mesh_verts_faces(0)

            if self.opt_settings['opt_geom']:
                verts3D = GenerateSphereSmoothVerts(xy = self.xy, z= self.z * self.z_scale[0], latitudes = self.latitudes, longitudes = self.longitudes)
                self.faces = self.mesh_faces
            # verts3D, self.faces  = self.mesh.get_mesh_verts_faces(0)
            verts3D[:, 0]  += self.mesh_centofrot[0]
            verts3D[:, 1]  += self.mesh_centofrot[1]
            verts3D[:, 2]  += self.mesh_centofrot[2]
            verts3D = Rotate(so3_exponential_map(self.meshLogR), device = self.device).transform_points(verts3D)
            # verts3D[:, 0]  =verts3D[:, 0].clone() *  self.mesh_scale[0]
            # verts3D[:, 1]  = verts3D[:, 1].clone() * self.mesh_scale[0]
            # verts3D[:, 2]  =  verts3D[:, 2].clone() *  self.mesh_scale[0]
            verts3D[:, 0]  += self.mesh_translate[0]
            verts3D[:, 1]  += self.mesh_translate[1]
            verts3D[:, 2]  += self.mesh_translate[2]
                
            # if self.opt_settings['opt_geom']:
            #     verts3D =  verts3D + (self.deform)
            #     self.mesh_opt =  Meshes(verts=[verts3D], faces=[self.faces])
            
            verts3D = verts3D.expand(self.views,-1,-1);
            self.faces = self.faces.expand(self.views,-1,-1);
            tex = None
            
            #add texture maps to meshes
            if self.mesh.textures is not None:
                faces_uvs = self.mesh.textures.faces_uvs_padded() 
                verts_uvs = self.mesh.textures.verts_uvs_padded()
                img = self.mesh.textures.maps_padded()
                img = torch.ones_like(img)
                tex = Textures(verts_uvs=verts_uvs.expand(len(view_batchids), -1, -1), faces_uvs=faces_uvs.expand(len(view_batchids), -1,-1), maps=img.expand(len(view_batchids), -1 , -1 ,-1))
           

        
        # Create a Meshes object to optimize with.
        meshes = Meshes(
            verts=verts3D[view_batchids,...], # batchsize of 1 for vertices  
            faces=self.faces[view_batchids,...], # batchsize of 1 for faces
            textures = tex,
        )
        self.meshes = meshes

        K = self.cameras.get_projection_transform().get_matrix()[view_batchids,...];
        focal_length = torch.zeros([len(view_batchids),2], device=self.device)
        principal_point = torch.zeros([len(view_batchids),2], device=self.device)
        focal_length[:,0] = K[:,0,0]
        focal_length[:,1] = K[:,1,1]
        principal_point[:,0] = K[:,0,2]
        principal_point[:,1] = K[:,1,2]
        cameras_batch = SfMPerspectiveCameras(device=self.device,
                                              focal_length = focal_length,
                                              principal_point = principal_point,
                                              R=self.cameras.R[view_batchids,...], 
                                              T=self.cameras.T[view_batchids,...]);
        ################################################################
        # create the renderer with a ReflectionMappingShader
        ################################################################
        shader_maps = {};
        shader_maps['diff_map'] = self.diff_map 
        shader_maps['spec_map'] = self.spec_map 
        shader_maps['gamma_map'] = self.gamma_map 
        shader_maps['amb_map'] = self.amb_map 
        
                # render the depthmap, images and normals and return
        mats = self.worldToScreenTransform.get_matrix()[view_batchids,...]
        batchScreenTrans = Transform3d(device=self.device)
        batchScreenTrans._matrix = mats
        
        ## render face reigon 

#         if self.face_mesh == None:
#           #For a purely specular reflection please use the following settings, 
#           #dif_map and amb_map should be 0, 
#           device = self.device
#           diff_map = 0.01*  torch.ones((1,1,3),device=device)
#           spec_map = 0.01*torch.ones((1,1,3),device=device)
#           gamma_map = 10000*torch.ones((1,1,3),device=device)
#           amb_map = 0*  torch.ones((1,1,3),device=device)
#           face_shader_maps=dict({'diff_map':diff_map, 'spec_map':spec_map,'gamma_map':gamma_map, 'amb_map':amb_map})
#           face_vert, face_faces =  self.face_mesh.get_mesh_verts_faces(0)
#           face_vert = Translate(x = self.mesh_translate[0], y = self.mesh_translate[1], z  = self.mesh_translate[2], device= self.device).transform_points(face_vert)
            
#           face_vert = face_vert.expand(self.views,-1,-1);
#           face_faces = face_faces.expand(self.views,-1,-1);
#           faces_uvs = self.face_mesh.textures.faces_uvs_padded() 
#           verts_uvs = self.face_mesh.textures.verts_uvs_padded()
#           img = self.face_mesh.textures.maps_padded()
#           tex = Textures(verts_uvs=verts_uvs.expand(len(view_batchids),-1,-1),faces_uvs=faces_uvs.expand(len(view_batchids), -1,-1),maps=img.expand(len(view_batchids),-1,-1,-1))
#           face_mesh = Meshes(
#             verts=face_vert[view_batchids,...], # batchsize of 1 for vertices  
#             faces=face_faces[view_batchids,...], # batchsize of 1 for faces
#             textures = tex
#           )
#           face_renderer = MeshRenderer(
#               rasterizer=MeshRasterizer(
#                   cameras=cameras_batch, 
#                   raster_settings=self.raster_settings
#               ),
#              shader=Tianfu_MultiviewReflectionMappingShader(cameras=cameras_batch, shader_maps=face_shader_maps, Np = self.Np, screen_size_u = self.screen_size_u, screen_size_v = self.screen_size_v, imheight = self.imheight, imwidth = self.imwidth)
#           )
#           _, face_normal, face_render, dist = face_renderer(meshes_world=face_mesh, screenTransform=batchScreenTrans, batch_pattern_maps = batch_texture_maps)   
        
        
        
        
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras_batch, 
                raster_settings=self.raster_settings
            ),
            shader=Tianfu_MultiviewReflectionMappingShader(cameras=cameras_batch, shader_maps=shader_maps, Np = self.Np, screen_size_u = self.screen_size_u, screen_size_v = self.screen_size_v, imheight = self.imheight, imwidth = self.imwidth)
        )
        

        depth_map, normal_map, images, dist , pixel_verts = renderer(meshes_world=meshes, screenTransform=batchScreenTrans, batch_pattern_maps = batch_texture_maps) 
        
        # print("images")
        # print(images.max())
#         ## add the face mask
#         if self.face_mesh == None:
#           face_mask = torch.ones_like(face_normal[:,:,:,0])
#           face_mask  = torch.sum(torch.abs(face_normal), axis = 3) <= 0
#           face_mask = face_mask.expand(3,-1,-1, -1).permute(1,2,3,0)
#           face_mask = face_mask.expand(len(texture_batchids), -1, -1, -1, -1)
#           # images = images * face_mask  +  face_render * (~face_mask )

        
        return depth_map, normal_map, images, dist, pixel_verts
    
    ################################################################
    # helper function to create the sinusoidal reflection maps 
    ################################################################
    def create_sinusoid_reflection_maps(self, Nt = [128, 128], Nph=4, periods=torch.tensor([1, 1])):
        
        ################################################################
        # create a batch of texture maps of phase shifted sinusoids
        #
        # inputs:
        # Nt - the number of pixels across in the sinusoidal texture map (default 128x128)
        # Nph- the number of phase shifts to use (default 4)
        # periods - (Nperiods, 2) array of sinusoid periods (default [1])
        #
        ################################################################
        #
        # returns:
        # textures - Textures object consisting of (2*Nph*Nperiods, Nt, Nt, 3) tensor
        # of RGB image textures, Nph*Nperiods are in X-direction, Nph*Nperiods are in Y-direction
        ###############################################################
        
        if isinstance(Nt, int):
            Nt_x, Nt_y = Nt, Nt
        else:
            Nt_x, Nt_y = Nt[0], Nt[1]
        # the number of periods to generate
        Nperiods = periods.shape[0];
        periods_x = torch.tensor(periods[:, 0:1], device=self.device, dtype=self.dtype)
        periods_x = periods_x.expand(Nph,Nt_x,Nt_y,Nperiods).permute(3,0,1,2).reshape(Nph*Nperiods, Nt_x, Nt_y) # Nph*Nperiods x Nt x Nt
        
        periods_y = torch.tensor(periods[:, 1:2], device=self.device, dtype=self.dtype)
        periods_y = periods_y.expand(Nph,Nt_x,Nt_y,Nperiods).permute(3,0,1,2).reshape(Nph*Nperiods, Nt_x, Nt_y) # Nph*Nperiods x Nt x Nt
        
        # the sampling grid for the sinusoids
        xp = np.linspace(-1,1,Nt_x) # grid is scaled -1,1
        yp = np.linspace(-1,1,Nt_y) # grid is scaled -1,1
        Ym,Xm = np.meshgrid(yp, xp);
        Ym = torch.tensor(Ym, device=self.device, dtype=self.dtype)
        Xm = torch.tensor(Xm, device=self.device, dtype=self.dtype)

        # create images for cosines in x/y directions
        t = torch.arange(0.,Nph, device=self.device, dtype=self.dtype)/Nph; # Nt - 1D coordinates in [0,1] range
        phase = 2*np.pi*t.expand(Nt_x, Nt_y,Nperiods, Nph).permute(2,3,0,1).reshape(Nph*Nperiods, Nt_x, Nt_y); # Nph*Nperiods x Nt x Nt
        Xarg = periods_x*np.pi*Xm.expand(Nph*Nperiods, Nt_x, Nt_y) + phase; # Nph*Nperiods x Nt x Nt
        Xcos = .5+.5*torch.cos(Xarg); # Np*Nperiods x Nt x Nt
        Yarg = periods_y*np.pi*Ym.expand(Nph*Nperiods, Nt_x, Nt_y) + phase; # Nph*Nperiods x Nt x Nt
        Ycos = .5+.5*torch.cos(Yarg); # Nph*Nperiods x Nt x Nt

        # concatenate X/Y texture maps and add 3 color channels
        texturemap = torch.cat((Xcos, Ycos), 0); # 2*Nph*Nperiods x Nt x Nt
        # expand to create three color channels
        texturemap = texturemap.expand(3,2*Nph*Nperiods,Nt_x,Nt_y) # (3, 2*Nph*Nperiods, Nt, Nt)

#         # expand to create three color channels
#         texturemap = Xcos.expand(3,Nph*Nperiods,Nt,Nt) # (3, 2*Nph*Nperiods, Nt, Nt)

        # texturemap: (3, 2*Nph*Nperiods, Nt, Nt) -> (2*Nph*Nperiods, 3, Nt, Nt) 
        texturemap = texturemap.permute(1,0,2,3)
        self.pattern = texturemap

    def get_angles(self):
        gaze = Rotate(so3_exponential_map(self.meshLogR), device = self.device).get_matrix()[0,2, :3]
        print(gaze.shape)
        azim = -torch.arctan(torch.tensor([gaze[0]/gaze[2]]))  * 180 /np.pi
        elev = torch.arctan(torch.tensor([gaze[1]/ torch.sqrt(gaze[2]* gaze[2] + gaze[0] * gaze[0]) ] ))  * 180 /np.pi
        
        return elev, azim, gaze
            

    def optimize_mesh(self, nits=200, lr=.001, its_disp=20, model_gt=None, views_batch=1, textures_batch=1, captured=None, \
                      sigma_diff=0, sigma_depth=0, view = 0, self_lr = False):
        # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
        
        ####################################################################################
        # normalize the learning rate by the maximum value of the parameters
        # assumption here is that learning rate is defined relative to maximum value of 1 for all parameters
        ####################################################################################

#         param_lr_dict = [{'params': param, 'lr': lr*self.max_height if name=='depthmap' 
#                           else lr*param.max().detach()} for name, param in self.named_parameters()]
        
#         param_lr_dict = [{'params': param, 'lr': lr*self.max_height if name=='depthmap' 
#                   else lr*param.max().detach()} for name, param in self.named_parameters()]
        
        # param_lr_dict = [{'params': param, 'lr':5e-3  if (name=='mesh_translate' or name =='xy')
        #           else lr*0.0080 } for name, param in self.named_parameters()]
        param_lr_dict = []
        for name, param in self.named_parameters():
            param_lr = 0.0080 * lr
            if name =='xy':
                param_lr = 1e-4
            if name == 'mesh_translate':
                param_lr = 1e-4
            if name == 'z_scale':
                param_lr = 5e-4
            if name == 'mesh_centofrot':
                param_lr = 1e-4

            
            param_lr_dict.append({'params': param, 'lr':param_lr })
            
        optimizer = torch.optim.Adam(param_lr_dict, lr=lr)
        print(param_lr_dict)
        
#         optimizer = torch.optim.Adam(self.parameters(), lr=lr)
#         optimizer = torch.optim.SGD(self.parameters(), lr=.001, momentum=0.2)

        # number of texturemaps
        
        # if self.logger is None:
        #     self.logger = SummaryWriter()
        n_textures = self.pattern.shape[0]
        saved_img = None
        save_view = 0
        save_view2 =1
        save_tex = 0
        if self.opt_settings['opt_mesh_transform'] or self.opt_settings['opt_pattern'] or self.opt_settings['opt_geom']:
            res = np.zeros(nits)
            res_history = torch.zeros((nits, 3)).to(self.device)
            losses = np.zeros(nits)
            print("Saving Ground truth view")
            if model_gt is not None:
                self.logger.add_image('GT_view_' + str(save_view) + '_texture_' + str(save_tex),(captured[save_tex, save_view]/torch.max(captured[save_tex, save_view])).clamp(min=0, max=0.99).permute(2,0,1), 0)
                self.logger.add_image('GT_texture_', model_gt.pattern[0].clamp(min=0, max=0.99), 0)
        if self.opt_settings['opt_pattern']:
            if model_gt is not None:
                self.logger.add_image('GT_texture_', model_gt.pattern[0].clamp(min=0, max=0.99), 0)
        for i in range(nits):
            # zero gradients
            loss = 0
            total_loss = 0
            
            for vb in range(self.views//views_batch):
                # render the images
                view_batchids = range(vb*views_batch, (vb+1)*views_batch)
                for tb in range(n_textures//textures_batch):
                    
                    optimizer.zero_grad()
                    loss = 0
                    tex_batchids = range(tb*textures_batch, (tb+1)*textures_batch)
                
                    
                    _,normal_opt,renders, _ , _= self.forward(view_batchids=view_batchids, texture_batchids=tex_batchids, \
                                               sigma_diff=sigma_diff, sigma_depth=sigma_depth)
                    # normal_opt = F.normalize(normal_opt, p=2, dim=-1, eps=1e-6)
                    normal_opt_corres = normal_opt[0, self.corres_idx_x[view_batchids[0]],self.corres_idx_y[view_batchids[0]],:]
                    # print("normal_opt")
                    # print(normal_opt.shape)
                    # print("normal_opt_corres")
                    # print(normal_opt_corres.shape)
                    normal_gt_corres = self.corres_norm[view_batchids[0]]
                    # print("normal_gt_corres")
                    # print(normal_gt_corres.shape)
                    # normal_gt_corres = normal_gt[view_batchids[0], corres_idx_x[view_batchids[0]],corres_idx_y[view_batchids[0]],:]
                    # loss = torch.mean(torch.log(torch.sum(torch.abs((normal_opt_corres - normal_gt_corres)), axis = -1) + 1 ))
                    loss = (torch.sum(torch.abs((normal_opt_corres - normal_gt_corres)) ** 2, axis = -1))
                    loss = torch.where (loss > 1, torch.log(loss) + 1, loss)
                    loss = torch.mean(loss)
                    

#                     renders = renders + 0.05 * torch.poisson(renders) + np.sqrt(0.1) * torch.randn_like(renders)
                    if model_gt is not None:
                        if (save_view in view_batchids) and (save_tex in tex_batchids):
                            save_img = renders[save_tex - tex_batchids[0], save_view - view_batchids[0]]    
                        if (save_view2 in view_batchids) and (save_tex in tex_batchids):
                            save_img2 = renders[save_tex - tex_batchids[0], save_view - view_batchids[0]] 
    #                     loss = ((renders - captured[tex_batchids,...][:,view_batchids,...]) [:,:,:,:, 2]).abs().mean() *len(view_batchids) * len(tex_batchids)


                    # if self.opt_settings['opt_geom']:
                    # Weight for mesh edge loss
                    w_edge = 0.01
                    # Weight for mesh normal consistency
                    w_normal = 0.01
                    # Weight for mesh laplacian smoothing
                    w_laplacian = 0.01

                    # and (b) the edge length of the predicted mesh
                    # loss_edge = mesh_edge_loss(self.meshes)

                    # mesh normal consistency
                    # loss_normal = mesh_normal_consistency(self.mesh_opt)

                    # mesh laplacian smoothing


                    # Weighted sum of the losses
                    # loss += loss_edge * w_edge + loss_laplacian * w_laplacian + loss_normal  * w_normal 
                                                
                        # verts = self.mesh_opt.verts_list()[0].detach().cpu().numpy()
    
                                                
                    # add tv loss on diff_map
                    if self.opt_settings['opt_diff']:
#                         tv = ((self.diff_map[0:-2,:,:] - self.diff_map[1:-1,:,:])**2).mean().sqrt() + \
#                              ((self.diff_map[:,0:-2,:] - self.diff_map[:,1:-1,:])**2).mean().sqrt()
                        tv = 1/self.diff_map.detach().max()*((self.diff_map[1:,:,:] - self.diff_map[:-1,:,:]).abs().mean() + \
                             (self.diff_map[:,1:,:] - self.diff_map[:,:-1,:]).abs().mean())

                        tv += 1/(self.depthmap.detach().max() + .000001)*((self.depthmap[:-1,:] - self.depthmap[1:,:]).abs().mean() + \
                              (self.depthmap[:,:-1] - self.depthmap[:,1:]).abs().mean())
                    if self.opt_settings['opt_geom']:
                    
                        xyz = torch.stack([self.xy,self.z *  self.z_scale[0]], axis = 1)
                        sign = torch.where(self.xy > 0, torch.zeros_like(self.xy),self.xy )
                        curvature = getCurvature(xyz[:-2], xyz[1:-1], xyz[2:]) - 4
                        curvature = torch.where(curvature < 0, torch.zeros_like(curvature),curvature )
                        verts3D = GenerateSphereSmoothVerts(xy = self.xy, z= self.z * self.z_scale[0], latitudes = self.latitudes, longitudes = self.longitudes)
                    
                        loss_laplacian = mesh_laplacian_smoothing(Meshes(verts=[verts3D ], faces=[self.mesh_faces]), method="uniform")
                        # print(f"lapal {loss_laplacian}")
                        print(self.xy,self.z)
                        print("curvature")
                        print(curvature)

                        gradient = (self.xy[:-1] - self.xy[1:])/((self.z[:-1] - self.z[1:]) *  self.z_scale[0])
                        gradient = torch.where(gradient > 0, torch.zeros_like(gradient),gradient )
                        print("gradient")
                        print(gradient)
                        
                        first_cons = torch.abs(self.xy[0] - 1)
                        last_cons = torch.abs(self.xy[-1])
                        # loss = loss -  0.0 * torch.sum(gradient) + 0.00* torch.sum (curvature) +  0 * (first_cons + last_cons) + 0.0 * loss_laplacian - 0.0 * torch.sum(sign)
                        loss = loss -  0.1 * torch.sum(gradient) + 0.01* torch.sum (curvature) +  0 * (first_cons + last_cons) + 0.1 * loss_laplacian - 0.0 * torch.sum(sign)
                        # loss = loss -  0.05 * torch.sum(gradient) + 0.001 * loss_laplacian - 0.0 * torch.sum(sign)
                    loss.backward(retain_graph=True)
                    # torch.nn.utils.clip_grad_norm_(self.parameters(), 5)

                    # update the parameters with gradient step
                    optimizer.step()
                    total_loss += float(loss)
                    
            self.logger.add_scalar('Loss', total_loss/self.views, i)
            losses[i] = total_loss

            if self.opt_settings['opt_mesh_transform']:
                if model_gt is not None:
                    rot_ang_opt = matrix_to_euler_angles(so3_exponential_map(self.meshLogR.detach()), "XYZ") * 180 /np.pi 
                    rot_ang_diff = rot_ang_opt - matrix_to_euler_angles(so3_exponential_map(model_gt.meshLogR), "XYZ")* 180 /np.pi 

                    R_err = so3_exponential_map(self.meshLogR.detach()).cpu().detach() - so3_exponential_map(model_gt.meshLogR.detach()).cpu().detach()
                    T_err = self.mesh_translate.cpu().detach() - model_gt.mesh_translate.cpu().detach()
                    T_err = torch.abs(T_err)
                    self.logger.add_scalars("Rotation_Angle_Opt", {"X":rot_ang_opt[0][0],"Y":rot_ang_opt[0][1],"Z":rot_ang_opt[0][2]}, i)
                    self.logger.add_scalars("Rotation_Angle_Dif", {"X":rot_ang_diff[0][0],"Y":rot_ang_diff[0][1],"Z":rot_ang_diff[0][2]}, i)
                    self.logger.add_scalars("Translate_Opt", {"X":self.mesh_translate[0],"Y":self.mesh_translate[1],"Z":self.mesh_translate[2]}, i)
                    self.logger.add_scalars("Translate_Dif", {"X":T_err[0],"Y":T_err[1],"Z":T_err[2]}, i)
                    self.logger.add_scalar('RMSE Rotation', R_err.std(), i)
                    self.logger.add_scalar('RMSE Translation', T_err.std(), i)
                    self.mesh_rotate_opt = Rotate(so3_exponential_map(self.meshLogR.cpu().detach()), device = self.device)
                    self.mesh_rotate_model = Rotate(so3_exponential_map(model_gt.meshLogR.cpu().detach()), device = self.device)
                    init_gaze = torch.tensor([[0.0, 0.0, 1.0]], device='cuda:0')
                    self.gaze_opt = self.mesh_rotate_opt.transform_normals(init_gaze)
                    self.gaze_gt = (self.mesh_rotate_model.transform_normals(init_gaze))
                    self.logger.add_scalar('Gazing_Dir_Angle', torch.arccos(self.gaze_opt[0].dot(self.gaze_gt[0])) * 180 /np.pi, i)
                    res[i] = torch.arccos(self.gaze_opt[0].dot(self.gaze_gt[0])) * 180 /np.pi
                    res_history[i] = Rotate(so3_exponential_map(self.meshLogR), device = self.device).get_matrix()[0,2, :3]
                    print('gazing direction angle' )
                    print(torch.arccos(self.gaze_opt[0].dot(self.gaze_gt[0])) * 180 /np.pi)
                    print("loss")
                    print(loss)
                    print(torch.abs(self.mesh_translate - model_gt.mesh_translate))
                # fig=plt.imshow(save_img.detach().cpu().numpy()/np.max(save_img.detach().cpu().numpy()))
                # plt.show()
                # fig=plt.imshow(save_img2.detach().cpu().numpy()/np.max(save_img2.detach().cpu().numpy()))
                # plt.show()

                # if i % its_disp == 0:
                    # fig=plt.imshow(save_img.detach().cpu().numpy()/np.max(save_img.detach().cpu().numpy()))
                    # plt.axis('off')
                    # fig.axes.get_xaxis().set_visible(False)
                    # fig.axes.get_yaxis().set_visible(False)
                    # plt.savefig(self.log_path + str(i) + '.png', bbox_inches = 'tight', pad_inches = 0)
     
#             if self.opt_settings['opt_pattern']:
#                 patten_err = self.pattern.detach().cpu().numpy()- model_gt.pattern.cpu().numpy()
#                 self.logger.add_scalar('Pattern error', patten_err.std(), i)
#                 if i % its_disp == 0:
#                     self.logger.add_image('view_' + str(save_view) + '_texture_' + str(save_tex), save_img, i)
#                     self.logger.add_image('Pattern', self.pattern.detach()[0].clamp(min=0, max=0.99).cpu().numpy(), i)
#             # if self.opt_settings['opt_geom']:
            #     if i % its_disp == 0:
            #         self.logger.add_image('view_' + str(save_view) + '_texture_' + str(save_tex), save_img, i)
                


            # Save outputs to create a GIF. 
            if i % its_disp == 0:
                print('Optimizing (loss %.4f)' % loss.data)
                if self.opt_settings['opt_diff']:
                    print('Diffuse map tv loss %.4f:' % tv.data)

                # display optimization progress of depthmap and normals
                if model_gt is not None:
                    self.display_progress(model_gt=model_gt, sigma_diff=sigma_diff, sigma_depth=sigma_depth)

        if self.opt_settings['opt_mesh_transform']:  
            return res_history, losses 
    def compare_renderings(self, rendered, captured):
        # reshape rendered images into views x Nph grid
        rendered = rendered.detach().cpu();
        captured = captured.detach().cpu();

        Nt,Nviews,_,_,_ = rendered.shape;

        # create interactive visualization of rendered images
        @interact(view=(0,Nviews-1),texture=(0,Nt-1))
        def viz(view,texture):
            f2 = plt.figure(figsize=(20, 10));
            plt.subplot(1,2,1)
            plt.title('rendered images for each view and screen texture')
            h = plt.imshow(rendered[texture,view,:,:,:].squeeze(), cmap="Greys_r", clim=(0, rendered.max())); 
            plt.subplot(1,2,2)
            plt.title('captured images for each view and screen texture')
            h = plt.imshow(captured[texture,view,:,:,:].squeeze(), cmap="Greys_r", clim=(0, rendered.max())); 

            plt.show()

                
    def visualize_renderings(self, depth=None, normals=None, images=None):
        # reshape rendered images into views x Nph grid
        renders = images.detach().cpu();
        Nt,Nviews,_,_,_ = renders.shape;

        if not (depth == None) and not (normals == None):
            # plot the depthmap estimate and error
            normals = normals[0,:,:,:].squeeze().cpu().detach().numpy()
            depth = depth.squeeze().cpu().detach().numpy()

            # plot the normalmap estimate and error
            f1 = plt.figure(figsize=(10, 5));
            plt.subplot(1,2,1)
            plt.imshow(.5*normals+.5);
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.imshow(depth/self.max_height);
            plt.colorbar()
            plt.show()

        # create interactive visualization of rendered images
        @interact(view=(0,Nviews-1),texture=(0,Nt-1))
        def viz(view,texture):
            f2 = plt.figure(figsize=(20, 10));
            plt.title('rendered images for each view and screen texture')
            h = plt.imshow(renders[texture,view,:,:,:].squeeze(), cmap="Greys_r", clim=(0, 1)); 
            plt.show()

    def visualize_textures(self):
        # load them and display
        tms = self.pattern.detach().cpu()
        tms = tms.permute(0,2,3,1);
        Nt,_,_,_ = tms.shape
        
        # create interactive visualization of textures
        @interact(t=(0,Nt-1))
        def viz(t):
            plt.title('sinusoidal texture maps to display on screen')
            h = plt.imshow(tms[t,:,:,:].squeeze(), cmap="Greys_r", clim=(0, 1)); 
            plt.show()    
            
    def display_progress(self, model_gt=None, sigma_diff=0, sigma_depth=0):
        
        if self.opt_settings['opt_mesh_transform']:
            R_err = so3_exponential_map(self.meshLogR).cpu().detach() - so3_exponential_map(model_gt.meshLogR).cpu().detach()
            T_err = self.mesh_translate.cpu().detach() - model_gt.mesh_translate.cpu().detach()
            print('RMSE Rotation (%.4f)' % R_err.std())
            print('RMSE Translation (%.4f)' % T_err.std())
            print("opt rotation in degrees")
            print(matrix_to_euler_angles(so3_exponential_map(self.meshLogR), "XYZ") * 180 /np.pi) 
            print("ground truth rotation in degrees")
            print(matrix_to_euler_angles(so3_exponential_map(model_gt.meshLogR), "XYZ") * 180 /np.pi) 
            print("opt translation")
            print(self.mesh_translate[0], self.mesh_translate[1], self.mesh_translate[2])
            print("ground truth translation")
            print(model_gt.mesh_translate)
        
    
        if self.opt_settings['opt_cam']:
            # visualize the camera positions
            self.cameras.R = so3_exponential_map(self.camLogR);
            self.cameras.T = self.camT;
            # f = plot_camera_scene(self.cameras, model_gt.cameras, "cameras")
            R_err = self.cameras.R.cpu().detach() - model_gt.cameras.R.cpu().detach()
            T_err = self.cameras.T.cpu().detach() - model_gt.cameras.T.cpu().detach()
            print('RMSE Rotation (%.4f)' % R_err.std())
            print('RMSE Translation (%.4f)' % T_err.std())
            
        if(self.opt_settings['opt_diff'] or self.opt_settings['opt_spec'] or 
           self.opt_settings['opt_gamma'] or self.opt_settings['opt_amb']):
            self.visualize_shader_maps(sigma_diff=sigma_diff)

            # blur the diff_map
            diff_map_opt = self.diff_map.detach()
            if sigma_diff>0:
                diff_map_opt = gaussian_blur_torch(diff_map_opt, sigma_diff)
            
            # get the shader maps for this object
            diff_map_opt = diff_map_opt.cpu().numpy()
            spec_map_opt = self.spec_map.cpu().detach().numpy()
            gamma_map_opt = self.gamma_map.cpu().detach().numpy()
            amb_map_opt = self.amb_map.cpu().detach().numpy()
            
            # get the shader maps for the ground truth object
            diff_map_gt = model_gt.diff_map.cpu().detach().numpy()
            spec_map_gt = model_gt.spec_map.cpu().detach().numpy()
            gamma_map_gt = model_gt.gamma_map.cpu().detach().numpy()
            amb_map_gt = model_gt.amb_map.cpu().detach().numpy()
    
            # get the error in estimates and print
            diff_map_err = (diff_map_opt-diff_map_gt);
            spec_map_err = (spec_map_opt-spec_map_gt);
            gamma_map_err = (gamma_map_opt-gamma_map_gt)
            amb_map_err = (amb_map_opt-amb_map_gt);
            print('Diffuse map error (%.4f)' % diff_map_err.std())
            print('Specular map error (%.4f)' % spec_map_err.std())
            print('Gamma map error (%.4f)' % gamma_map_err.std())
            print('Ambient map error (%.4f)' % amb_map_err.std())

    def visualize_shader_maps(self, sigma_diff=0):
        diff_map = self.diff_map
        spec_map = self.spec_map
        gamma_map = self.gamma_map
        amb_map = self.amb_map

        if sigma_diff>0:
            diff_map = gaussian_blur_torch(diff_map, sigma_diff)

        # view the shader maps
        f1 = plt.figure(figsize=(10, 10));
        plt.subplot(2,2,1)
        plt.title('diffuse map')
        plt.imshow(diff_map.cpu().detach()/diff_map.cpu().detach().max());
#         plt.clim(diff_map.min(), diff_map.max())
        plt.colorbar()

        plt.subplot(2,2,2)
        plt.title('specular map')
        plt.imshow(spec_map.cpu().detach());
        plt.clim(spec_map.min(), spec_map.max())
        plt.colorbar()

        plt.subplot(2,2,3)
        plt.title('gamma map')
        plt.imshow(gamma_map.cpu().detach());
        plt.clim(gamma_map.min(), gamma_map.max())
        plt.colorbar()

        plt.subplot(2,2,4)
        plt.title('ambient map')
        plt.imshow(amb_map.cpu().detach());
        plt.clim(amb_map.min(), amb_map.max())
        plt.colorbar()

        plt.show()

        
def GenerateSphereSmoothVerts(xy = None, z = None, radius = 0.01185, latitudes = 100, longitudes = 100, dtype = torch.float32, device = "cuda:0"):
    verts = torch.zeros(((latitudes + 1) * (longitudes + 1), 3), dtype= dtype,device = device)
    
    
    latitudeAngles = torch.linspace(np.pi/2, -np.pi/2, latitudes + 1, dtype= dtype,device = device)
    # z = torch.linspace(-1, 1, latitudes + 1, dtype= dtype,device = device)
    z_radius = radius * torch.sin(latitudeAngles)
    xy_radius = torch.sqrt(radius * radius - z_radius * z_radius)
    if z != None:
        z_radius[latitudes//2:] = radius* z
    # z[10:] = torch.linspace(0, -1, 55, dtype= dtype,device = device)
    # xy = torch.sqrt(radius * radius * torch.ones_like(z) - z * z)
    # xy[latitudes//2:] = torch.linspace(xy[latitudes//2], 0, latitudes + 1 - latitudes//2, dtype= dtype,device = device)
    if xy != None:        
        xy_radius[latitudes//2:] = radius * xy
    logitudeAngles = torch.linspace(0, 2 * np.pi, longitudes + 1, dtype= dtype,device = device)
    x = torch.outer(xy_radius, torch.cos(logitudeAngles))
    x = x.flatten()
    y = torch.outer(xy_radius, torch.sin(logitudeAngles)).flatten()
    z_radius = z_radius.repeat_interleave(longitudes + 1)
    
    verts[:, 0] = x
    verts[:, 1] = y
    verts[:, 2] = z_radius
    return verts

def GenerateSphereSmoothFaces(latitudes = 100, longitudes = 100, dtype = torch.int64, device = "cuda:0"):
    k1 = 0 
    k2 = 0
    faces = torch.zeros(( 2 * (latitudes  - 1) * (longitudes), 3), dtype= dtype,device = device)
    offset = 0
    for i in range(latitudes):
        k1 = i * (longitudes + 1);
        k2 = k1 + longitudes + 1;
        for j in range(longitudes):
            if (i != 0):
                faces[offset, 0] = k1
                faces[offset, 1] = k2
                faces[offset, 2] = k1 + 1
                offset =offset + 1
            if (i != (latitudes - 1)):
                faces[offset, 0] = k1 + 1
                faces[offset, 1] = k2
                faces[offset, 2] = k2 + 1
                offset = offset + 1
            k1 = k1 + 1
            k2 = k2 + 1
    return faces

def getCurvature(pa, pb, pc):
    '''
    calculates the curvature of a triangle formed by three points (pa, pb, pc), using Menger's curvature formula,
    returns this curvature value
    '''

    #https://en.wikipedia.org/wiki/Menger_curvature
    ab = torch.sqrt(torch.sum((pa - pb) * (pa - pb), axis = 1))
    bc = torch.sqrt(torch.sum((pb - pc) * (pb - pc), axis = 1))
    ca = torch.sqrt(torch.sum((pc - pa) * (pc - pa), axis = 1))

    #https://www.mathopenref.com/coordtrianglearea.html
    area = 0.5 * torch.abs(pa[:, 1] * (pb[:, 0]  - pc[:, 0])  
                     + pb[:, 1] * (pc[:, 0]  - pa[:, 0]) 
                     + pc[:, 1] * (pa[:, 0]  - pb[:, 0]) ) 
    print("curve multi")
    print(ab * bc * ca)
    
    return 4 * area /(ab * bc * ca)
