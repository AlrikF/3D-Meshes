import trimesh
import matplotlib.pyplot as plt
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
    def __init__(self,he_name, origin, twin_name, incident_face, next, prev):
        self.he_name = he_name
        self.origin = origin
        self.twin_name = twin_name
        self.incident_face = incident_face
        self.next = next
        self.prev = prev

    def __str__(self):  
        return f" half edge: {self.he_name}, origin vertice: {self.origin}, twin: {self.twin_name}, incident face: {self.incident_face} \
            ,next: {self.next}, prev: {self.prev }"
    
    def __str__(self):  
        return f" half edge: {self.he_name}, origin vertice: {self.origin}, twin: {self.twin_name}, incident face: {self.incident_face} \
            ,next: {self.next}, prev: {self.prev }"




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

    
    print(mesh.vertices)

    vertices_list = {}
    for i in range(len(mesh.vertices)): 
        vertices_list[f"v{i}"]= Vertice_Edge_Node(f"v{i}",mesh.vertices[i],None)  

    faces_list = {}
    for i in range(len(mesh))

    print(vertices_list)

    # TODO: implement your own loop subdivision here
    mesh_subdivided = subdivision_loop(mesh, iterations=1)
    
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