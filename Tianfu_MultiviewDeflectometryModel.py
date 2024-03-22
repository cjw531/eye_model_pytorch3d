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

from Tianfu_MultiviewReflectionMappingShader import Tianfu_MultiviewReflectionMappingShader 

def gaussian_blur_torch(in_tensor, sigma=0):
    '''
    Apply Gaussian blur to a 2D tensor
    '''

    if sigma > 0:
        # create Gaussian matrix
        kernel_width = 4 * int(np.ceil(sigma)) + 1
        xp = np.linspace(-kernel_width / 2, kernel_width / 2, kernel_width) # grid is scaled -1,1
        Y, X = np.meshgrid(xp, xp)
        G = np.exp(-(X**2 + Y**2) / 2 / sigma)[None, None, ...] # 2D Gaussian distribution
        G = torch.tensor(G, dtype=in_tensor.dtype).to(in_tensor.device) # conver to PyTorch tensor
        G = G/G.sum() # normalize to sum to 1
        
        # create Gaussian weights for each color channel
        if in_tensor.dim() < 3:
            in_tensor = in_tensor[...,None]
        H, W, C = in_tensor.shape

        w = torch.zeros((C, C, kernel_width, kernel_width), dtype=in_tensor.dtype).to(in_tensor.device)
        for c in range(C):
            w[c, c, : , :] = G

        # blur the input tensor
        blurred = F.conv2d(in_tensor.permute(2,0,1)[None,...], weight=w, padding=kernel_width//2).permute(0,2,3,1).squeeze()
        
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
        self.dtype=dtype
        self.device=device
        self.max_height = max_height
        self.Nv = Nv; # number of vertices in mesh is Nv**2
        # start with empty normals
        self.normals = torch.zeros((Np,Np,3), dtype=dtype)
        self.opt_settings = opt_settings
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
        self.shader_maps = shader_maps
        
        ################################################################
        # initialize mesh 
        ################################################################
        # if there is no depthmap provided, initialize to zeros
        if self.mesh == None:
            if init_depth == None:
                self.Nv = 2
                depthmap = torch.zeros((Nv,Nv, 1), dtype=dtype)
            else:
                self.Nv = init_depth.shape[0]
            self.depthmap = init_depth

            if init_verts2d == None:
                # create a regular grid of spatial coords for uniform mesh
                xp = np.linspace(-1,1,Nv) # grid is scaled -1,1
                Y,X = np.meshgrid(xp, xp)
    
                # create vertices and faces
                init_verts2d = torch.cat((torch.tensor(Y, dtype=dtype).view(Nv*Nv,1), 
                                     torch.tensor(X, dtype=dtype).view(Nv*Nv,1)), 1).to(device) # (1, Nv*Nv, 2)
            self.verts2D = init_verts2d
    
            # use scipy to help with Delaunay triangulation of the faces
            tri = sp.Delaunay(self.verts2D.cpu().numpy())
            self.faces = torch.tensor(tri.simplices, dtype=torch.int64).to(device); # (1, F, 3) - datatype must be integer for indices
        else:
            _, self.faces  = self.mesh.get_mesh_verts_faces(0)

        ################################################################
        # choose initial camera model 
        ################################################################
        if init_cam == None:
            self.views = 1
            # Get the position of the camera based on the spherical angles
            R, T = look_at_view_transform(dist=2, elev=0, azim=0, device=device) # (views,3,3), (views,3) 
            init_cam = SfMPerspectiveCameras(device=device, R=R, T=T)
        
        self.cameras = init_cam
        
        # optimize the rotation in axis-angle format
        self.camT = self.cameras.T
        self.camLogR = so3_log_map(self.cameras.R); #(views, 3) - want to optimize axis-angle
        self.camLogR[torch.isnan(self.camLogR)] = 0; # set nans to zero

        ################################################################
        # create the screen transforms 
        ################################################################
        if cameraToScreenTransform == None:
            if screenT==None:
                screenT = self.cameras.T
            if screenR==None:
                screenR = self.cameras.R
            if screenS==None:
                screenScale = torch.tensor(1)

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
                self.mesh_translate = nn.Parameter(self.mesh_translate)
            if opt_rotate:
                self.meshLogR = nn.Parameter(self.meshLogR)
            if opt_centofrot:
                self.mesh_centofrot = nn.Parameter(self.mesh_centofrot)
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
            self.latitudes = latitudes
            self.longitudes  = longitudes
            delta_latitude = np.pi /latitudes
            delta_logitude = 2 * np.pi/longitudes
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
        self.views = self.cameras.R.shape[0]
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
            mask = self.fixed_depthmap[:,:,0:1]
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
        shader_maps = {}
        shader_maps['diff_map'] = self.diff_map 
        shader_maps['spec_map'] = self.spec_map 
        shader_maps['gamma_map'] = self.gamma_map 
        shader_maps['amb_map'] = self.amb_map 
        
        # render the depthmap, images and normals and return
        mats = self.worldToScreenTransform.get_matrix()[view_batchids,...]
        batchScreenTrans = Transform3d(device=self.device)
        batchScreenTrans._matrix = mats
        
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras_batch, 
                raster_settings=self.raster_settings
            ),
            shader=Tianfu_MultiviewReflectionMappingShader(cameras=cameras_batch, shader_maps=shader_maps, Np = self.Np, screen_size_u = self.screen_size_u, screen_size_v = self.screen_size_v, imheight = self.imheight, imwidth = self.imwidth)
        )
        
        depth_map, normal_map, images, dist , pixel_verts = renderer(meshes_world=meshes, screenTransform=batchScreenTrans, batch_pattern_maps = batch_texture_maps) 
        
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
        Nperiods = periods.shape[0]
        periods_x = torch.tensor(periods[:, 0:1], device=self.device, dtype=self.dtype)
        periods_x = periods_x.expand(Nph,Nt_x,Nt_y,Nperiods).permute(3,0,1,2).reshape(Nph*Nperiods, Nt_x, Nt_y) # Nph*Nperiods x Nt x Nt
        
        periods_y = torch.tensor(periods[:, 1:2], device=self.device, dtype=self.dtype)
        periods_y = periods_y.expand(Nph,Nt_x,Nt_y,Nperiods).permute(3,0,1,2).reshape(Nph*Nperiods, Nt_x, Nt_y) # Nph*Nperiods x Nt x Nt
        
        # the sampling grid for the sinusoids
        xp = np.linspace(-1,1,Nt_x) # grid is scaled -1,1
        yp = np.linspace(-1,1,Nt_y) # grid is scaled -1,1
        Ym,Xm = np.meshgrid(yp, xp)
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
        texturemap = texturemap.permute(1,0,2,3)
        self.pattern = texturemap
                
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
        tms = tms.permute(0,2,3,1)
        Nt,_,_,_ = tms.shape
        
        # create interactive visualization of textures
        @interact(t=(0,Nt-1))
        def viz(t):
            plt.title('sinusoidal texture maps to display on screen')
            h = plt.imshow(tms[t,:,:,:].squeeze(), cmap="Greys_r", clim=(0, 1)); 
            plt.show()

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
    '''
    Generate vertices
    '''

    verts = torch.zeros(((latitudes + 1) * (longitudes + 1), 3), dtype= dtype,device = device) # sphere vertices
    
    latitudeAngles = torch.linspace(np.pi/2, -np.pi/2, latitudes + 1, dtype=dtype, device=device)
    z_radius = radius * torch.sin(latitudeAngles)
    xy_radius = torch.sqrt(radius * radius - z_radius * z_radius)
    if z != None:
        z_radius[latitudes//2:] = radius* z
    if xy != None:        
        xy_radius[latitudes//2:] = radius * xy
    logitudeAngles = torch.linspace(0, 2 * np.pi, longitudes + 1, dtype=dtype, device=device)

    x = torch.outer(xy_radius, torch.cos(logitudeAngles))
    x = x.flatten()
    y = torch.outer(xy_radius, torch.sin(logitudeAngles)).flatten()
    z_radius = z_radius.repeat_interleave(longitudes + 1)
    
    verts[:, 0] = x
    verts[:, 1] = y
    verts[:, 2] = z_radius

    return verts

def GenerateSphereSmoothFaces(latitudes = 100, longitudes = 100, dtype = torch.int64, device = "cuda:0"):
    '''
    Generate smooth faces of a smooth sphere
    '''

    k1, k2 = 0, 0
    offset = 0
    faces = torch.zeros((2 * (latitudes  - 1) * (longitudes), 3), dtype=dtype, device=device)
    
    for i in range(latitudes):
        k1 = i * (longitudes + 1)
        k2 = k1 + longitudes + 1

        for j in range(longitudes):
            if (i != 0):
                faces[offset, 0] = k1
                faces[offset, 1] = k2
                faces[offset, 2] = k1 + 1
                offset += 1
            if (i != (latitudes - 1)):
                faces[offset, 0] = k1 + 1
                faces[offset, 1] = k2
                faces[offset, 2] = k2 + 1
                offset += 1
            k1 += 1
            k2 += 1

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
