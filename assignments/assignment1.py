import trimesh
import matplotlib.pyplot as plt
import numpy as np

class Vertice_Edge_Node:
    def __init__(self,name, coordinates, incident_edge):
        self.vname= name
        self.coordinates= coordinates
        self.incident_edge= incident_edge

    
    def __str__(self):  
        return f" vertice : {self.vname},  coordinates: {self.coordinates}, incident: {self.incident_edge}"
    
    def __repr__(self):  
        return f" vertice : {self.vname},  coordinates: {self.coordinates}, incident: {self.incident_edge}"

class Face_Node:
    def __init__(self,name, half_edge):
        self.fname = name
        self.half_edge= half_edge

    def __str__(self):  
        return f" face : {self.fname},  half edge: {self.half_edge}"
    
    def __str__(self):  
        return f" face : {self.fname},  half edge: {self.half_edge}"

    

class Half_Edge_Node:
    def __init__(self,he_name, origin,end_vertex= None, twin_name=None, incident_face=None, next=None, prev=None):
        self.he_name = he_name
        self.origin = origin
        self.end_vertex = end_vertex
        self.twin_name = twin_name
        self.incident_face = incident_face
        self.next = next
        self.prev = prev

    def __str__(self):  
        return f"\n half edge: {self.he_name}, origin vertice: {self.origin}, twin: {self.twin_name}, incident face: {self.incident_face}, next: {self.next}, prev: {self.prev } |\n"
    
    def __repr__(self):  
        return f"\n half edge: {self.he_name}, origin vertice: {self.origin}, twin: {self.twin_name}, incident face: {self.incident_face}, next: {self.next}, prev: {self.prev } |\n"


def create_half_edge(mesh, vertices_list ,half_edge_list, faces_list):
    for i in range(len(mesh.vertices)): 
        vertices_list[f"v{i}"]= Vertice_Edge_Node(f"v{i}",np.array(mesh.vertices[i]),None)  

    # Step 2 & 3: Create half-edges
    edge_count=0
    for i in range(len(mesh.triangles)):
        triangle= mesh.triangles[i]
        faces_list[f"f{i}"]=Face_Node(f"f{i}", None)
        
        print("Triangle:",triangle)
        
        for j in range(3):
            v_name=None
            for pt in vertices_list:
                # print(vertices_list[pt].coordinates , triangle[j], type(vertices_list[pt].coordinates) ,type(triangle[j]))
                if np.array_equal(vertices_list[pt].coordinates, triangle[j]):
                    v_name = pt 
                    break
            if v_name==None:
                print("Vertex not foud for coordinates",triangle[j])
            
            half_edge_list[f"e{edge_count}"] = Half_Edge_Node(f"e{edge_count}",v_name)
            half_edge_list[f"e{edge_count}"].incident_face = f"f{i}"
            
            # Assign half edge to a face 
            if j == 0:
                faces_list[f"f{i}"].half_edge = f"e{edge_count}"
            if j > 0:
                half_edge_list[f"e{edge_count}"].prev = f"e{edge_count-1}"
                half_edge_list[f"e{edge_count-1}"].next = f"e{edge_count}"
                half_edge_list[f"e{edge_count-1}"].end_vertex = half_edge_list[f"e{edge_count}"].origin
            if j == 2:
                half_edge_list[f"e{edge_count}"].next = f"e{edge_count-2}"
                half_edge_list[f"e{edge_count-2}"].prev = f"e{edge_count}"
                half_edge_list[f"e{edge_count}"].end_vertex = half_edge_list[f"e{edge_count-2}"].origin

            edge_count+=1

    # Step 4: Connect half-edges
    for he_name in half_edge_list:
        if half_edge_list[he_name].twin_name is None:
            for twin_he_name in half_edge_list:
                if (half_edge_list[twin_he_name].origin == half_edge_list[he_name].end_vertex and
                        half_edge_list[twin_he_name].end_vertex == half_edge_list[he_name].origin):
                    half_edge_list[he_name].twin_name = twin_he_name
                    half_edge_list[twin_he_name].twin_name = he_name 
                    break


def subdivision_loop(mesh, iterations=1):
    """
    Apply Loop subdivision to the input mesh for the specified number of iterations.
    :param mesh: input mesh
    :param iterations: number of iterations
    :return: mesh after subdivision
    """
    mesh.show(flags={'wireframe': True,"vertices":True})

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')


    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces);

    plt.show()
    return mesh

def simplify_quadric_error(mesh, face_count=1):
    """
    Apply quadratic error mesh decimation to the input mesh until the target face count is reached.
    :param mesh: input mesh
    :param face_count: number of faces desired in the resulting mesh.
    :return: mesh after decimation
    """

    # Get vertices of the mesh
    vertices = mesh.vertices

    # Convert vertices to a list
    coordinates = vertices.tolist()
    print(coordinates)


    return mesh

if __name__ == '__main__':
    # Load mesh and print information
    # mesh = trimesh.load_mesh('assets/cube.obj')
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    print(f'Mesh Info: {mesh}')
    
    # apply loop subdivision over the loaded mesh
    mesh_subdivided = mesh.subdivide_loop(iterations=1)

    
    print("Vertices :", mesh.vertices)

    vertices_list = {}
    

    half_edge_list={}

    faces_list = {}

    create_half_edge(mesh=mesh,vertices_list=vertices_list, half_edge_list=half_edge_list, faces_list=faces_list)

    print(half_edge_list)
    
    

    print(vertices_list)

    # TODO: implement your own loop subdivision here
    # mesh_subdivided = subdivision_loop(mesh, iterations=1)
    
   
   
   
   
   
   
   
   
    # print the new mesh information and save the mesh
    # print(f'Subdivided Mesh Info: {mesh_subdivided}')
    # mesh_subdivided.export('assets/assignment1/cube_subdivided.obj')
    
    # # quadratic error mesh decimation
    # mesh_decimated = mesh.simplify_quadric_decimation(4)
    
    # # TODO: implement your own quadratic error mesh decimation here
    # # mesh_decimated = simplify_quadric_error(mesh, face_count=1)
    
    # # print the new mesh information and save the mesh
    # print(f'Decimated Mesh Info: {mesh_decimated}')
    # mesh_decimated.export('assets/assignment1/cube_decimated.obj')