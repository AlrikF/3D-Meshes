import trimesh
import matplotlib.pyplot as plt
import numpy as np
import copy

class Vertice_Edge_Node:
    def __init__(self,name, coordinates, incident_edge=None):
        self.vname= name
        self.coordinates= coordinates
        self.incident_edge= incident_edge

    
    def __str__(self):  
        return f"\n vertice : {self.vname},  coordinates: {self.coordinates}, incident: {self.incident_edge} |\n"
    
    def __repr__(self):  
        return f"\n vertice : {self.vname},  coordinates: {self.coordinates}, incident: {self.incident_edge} |\n"

class Face_Node:
    def __init__(self,name, half_edge):
        self.fname = name
        self.half_edge= half_edge

    def __str__(self):  
        return f"\n face : {self.fname},  half edge: {self.half_edge} |\n"
    
    def __repr__(self):  
        return f"\n face : {self.fname},  half edge: {self.half_edge} |\n"

    

class Half_Edge_Node:
    def __init__(self,he_name, origin,end_vertex= None, twin_name=None, incident_face=None, next=None, prev=None):
        self.he_name = he_name
        self.origin = origin
        self.end_vertex = end_vertex
        self.twin_name = twin_name
        self.incident_face = incident_face
        self.next = next
        self.prev = prev
        self.child_odd_vertex = None

    def __str__(self):  
        return f"\n half edge: {self.he_name}, origin vertice: {self.origin}, end vertex: {self.end_vertex}, twin: {self.twin_name}, incident face: {self.incident_face}, next: {self.next}, prev: {self.prev }, mid: {self.child_odd_vertex} |\n"
    
    def __repr__(self):  
        return f"\n half edge: {self.he_name}, origin vertice: {self.origin}, end vertex: {self.end_vertex}, twin: {self.twin_name}, incident face: {self.incident_face}, next: {self.next}, prev: {self.prev }, mid: {self.child_odd_vertex} |\n"




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

def plot_half_edge(new_half_edge_list, new_vertices_list):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Iterate through half edges and plot them
    for e_name in new_half_edge_list:
        org, end = new_half_edge_list[e_name].origin, new_half_edge_list[e_name].end_vertex
        org_coords, end_coords = new_vertices_list[org].coordinates, new_vertices_list[end].coordinates
        ax.plot([org_coords[0], end_coords[0]], [org_coords[1], end_coords[1]], [org_coords[2], end_coords[2]], color='b')

    # Set labels and show plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
          

    # Annotating vertices
    for vertex_name in vertices_list:
        ax.text(vertices_list[vertex_name].coordinates[0], vertices_list[vertex_name].coordinates[1], vertices_list[vertex_name].coordinates[2], vertex_name)

        
    for vertex_name in new_vertices_list:
        ax.scatter(new_vertices_list[vertex_name].coordinates[0], new_vertices_list[vertex_name].coordinates[1], new_vertices_list[vertex_name].coordinates[2], color='r')
        ax.text(new_vertices_list[vertex_name].coordinates[0], new_vertices_list[vertex_name].coordinates[1], new_vertices_list[vertex_name].coordinates[2], vertex_name)
        

    plt.show()

def compute_odd_vertices(mesh,half_edge_list,vertices_list,new_vertices_list):
    visited=set([])
    for e_name in half_edge_list:
        if e_name not in visited:
            print(e_name,visited)
            twin_e_name=half_edge_list[e_name].twin_name
            visited.add(e_name)
            visited.add(twin_e_name)

            higher_weight_vertex1 = half_edge_list[e_name].origin
            higher_weight_vertex2 = half_edge_list[e_name].end_vertex

            lower_weight_vertex1 = half_edge_list[half_edge_list[e_name].next].end_vertex
            lower_weight_vertex2 = half_edge_list[half_edge_list[twin_e_name].next].end_vertex

            odd_vertex= 3/8*(vertices_list[higher_weight_vertex1].coordinates +vertices_list[higher_weight_vertex2].coordinates) \
                       +1/8*(vertices_list[lower_weight_vertex1].coordinates + vertices_list[lower_weight_vertex2].coordinates)
            

            half_edge_list[e_name].child_odd_vertex = f"v{len(new_vertices_list)}"
            half_edge_list[twin_e_name].child_odd_vertex = f"v{len(new_vertices_list)}"
            new_vertices_list[f"v{len(new_vertices_list)}"] = Vertice_Edge_Node(f"v{len(new_vertices_list)}",np.array(odd_vertex))




def compute_even_vertices(mesh,half_edge_list,vertices_list, new_vertices_list):
    
    for vertex_name in vertices_list:
        neighbours = set([])
        for he_name in half_edge_list:
            if half_edge_list[he_name].origin == vertex_name:
                neighbours.add(half_edge_list[he_name].end_vertex)
        sum_neighbours = 0
        for neighbour in neighbours:
            sum_neighbours+= vertices_list[neighbour].coordinates
        
        print("Sum neighbours :", sum_neighbours)

        beta=3/16.0
        if len(neighbours)>3:
            beta=3/(8*len(neighbours))
        elif len(neighbours)<3:
            raise Warning(f"Too few neighbours for {vertex_name} :: {neighbours}")

        new_vertices_list[vertex_name].coordinates = vertices_list[vertex_name].coordinates*(1-len(neighbours)*beta) +sum_neighbours*beta


def create_new_faces(new_face_list,new_half_edge_list, triangle):
    curr_face= f"f{len(new_face_list)}"
    edge_count = len(new_half_edge_list)
    for j in range(3):
        
        new_half_edge_list[f"e{edge_count}"] = Half_Edge_Node(f"e{edge_count}",triangle[j])
        new_half_edge_list[f"e{edge_count}"].incident_face = curr_face
        
        # Assign half edge to a face 
        if j == 0:
            new_face_list[curr_face]=Face_Node(curr_face,new_half_edge_list[f"e{edge_count}"])
            new_face_list[curr_face].half_edge = f"e{edge_count}"
        if j > 0:
            new_half_edge_list[f"e{edge_count}"].prev = f"e{edge_count-1}"
            new_half_edge_list[f"e{edge_count-1}"].next = f"e{edge_count}"
            new_half_edge_list[f"e{edge_count-1}"].end_vertex = new_half_edge_list[f"e{edge_count}"].origin
        if j == 2:
            new_half_edge_list[f"e{edge_count}"].next = f"e{edge_count-2}"
            new_half_edge_list[f"e{edge_count-2}"].prev = f"e{edge_count}"
            new_half_edge_list[f"e{edge_count}"].end_vertex = new_half_edge_list[f"e{edge_count-2}"].origin

        edge_count+=1

    




def compute_new_faces(faces_list, new_vertices_list,new_half_edge_list,new_face_list):
    
    for f_name in faces_list:
        he_name= faces_list[f_name].half_edge
        prev_he_name= half_edge_list[he_name].prev
        next_he_name= half_edge_list[he_name].next
        
        origin_vertex1 = half_edge_list[he_name].origin    
        end_vertex1 = half_edge_list[he_name].end_vertex
        child_odd_vertex1 = half_edge_list[he_name].child_odd_vertex

        origin_vertex2 = half_edge_list[prev_he_name].origin    
        end_vertex2 = half_edge_list[prev_he_name].end_vertex
        child_odd_vertex2 = half_edge_list[prev_he_name].child_odd_vertex

        origin_vertex3 = half_edge_list[next_he_name].origin    
        end_vertex3 = half_edge_list[next_he_name].end_vertex
        child_odd_vertex3 = half_edge_list[he_name].child_odd_vertex

        # print("Creating 4 new faces")
        # print(new_face_list)
        print(origin_vertex1,child_odd_vertex1,child_odd_vertex2)
        print(origin_vertex2,child_odd_vertex3,child_odd_vertex2)
        print(origin_vertex3,child_odd_vertex1,child_odd_vertex3)
        print(child_odd_vertex3,child_odd_vertex1,child_odd_vertex2)


        create_new_faces(new_face_list,new_half_edge_list,[origin_vertex1,child_odd_vertex1,child_odd_vertex2])
        create_new_faces(new_face_list,new_half_edge_list,[origin_vertex2,child_odd_vertex3,child_odd_vertex2])
        create_new_faces(new_face_list,new_half_edge_list,[origin_vertex3,child_odd_vertex1,child_odd_vertex3])
        create_new_faces(new_face_list,new_half_edge_list,[child_odd_vertex3,child_odd_vertex1,child_odd_vertex2])
        # print(new_face_list)



def subdivision_loop(mesh, vertices_list, half_edge_list, faces_list, iterations=1):
    """
    Apply Loop subdivision to the input mesh for the specified number of iterations.
    :param mesh: input mesh
    :param iterations: number of iterations
    :return: mesh after subdivision
    """
    new_vertices_list = copy.deepcopy(vertices_list)
    new_half_edge_list = {}
    new_face_list = {}
    for _ in range(iterations):
        # new_half_edge_list= copy.deepcopy(half_edge_list)
        new_vertices_list = copy.deepcopy(vertices_list)
        new_half_edge_list = {}
        new_face_list = {}
        # Step 1: Compute updated positions for existing vertices

        compute_odd_vertices(mesh,half_edge_list=half_edge_list,vertices_list=vertices_list,new_vertices_list=new_vertices_list )
      
        compute_even_vertices(mesh,half_edge_list=half_edge_list,vertices_list=vertices_list,new_vertices_list=new_vertices_list)
        
        compute_new_faces(faces_list, new_vertices_list,new_half_edge_list,new_face_list )
        
        # Step 4: Connect half-edges
        for he_name in new_half_edge_list:
            if new_half_edge_list[he_name].twin_name is None:
                for twin_he_name in new_half_edge_list:
                    if (new_half_edge_list[twin_he_name].origin == new_half_edge_list[he_name].end_vertex and
                        new_half_edge_list[twin_he_name].end_vertex == new_half_edge_list[he_name].origin):
                        new_half_edge_list[he_name].twin_name = twin_he_name
                        new_half_edge_list[twin_he_name].twin_name = he_name 
                        break
        
        half_edge_list= copy.deepcopy(new_half_edge_list)
        faces_list = copy.deepcopy(new_face_list)
        vertices_list = copy.deepcopy(new_vertices_list)

    print(len(new_half_edge_list),len(new_vertices_list),len(new_face_list)) 
    plot_half_edge(new_half_edge_list,new_vertices_list)
    

    
    return new_half_edge_list, new_vertices_list,new_face_list

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



def write_obj_file(vertices_list, face_list, filename):
    with open(filename, 'w') as obj_file:
        # Write vertex coordinates
        for vertex_name, vertex_data in vertices_list.items():
            obj_file.write(f"v {' '.join(map(str, vertex_data.coordinates))}\n")

        # Write faces
        for face_node in face_list.values():
            half_edge = new_half_edge_list[face_node.half_edge]
            face_vertices = [half_edge.origin]
            next_edge = half_edge.next
            while next_edge != face_node.half_edge:
                half_edge = new_half_edge_list[next_edge]
                face_vertices.append(half_edge.origin)
                next_edge = half_edge.next
            obj_file.write("f " + " ".join(face_vertices) + "\n")



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

    print("\n Half Edge List ::\n", half_edge_list)

    
    print(" \n Faces List : ")

    print("\n Vertices List \n",vertices_list)

   

    # TODO: implement your own loop subdivision here
    new_half_edge_list, new_vertices_list, new_face_list = subdivision_loop(mesh, vertices_list, half_edge_list, faces_list, iterations=2)
    
    print("\n New Vertices List \n",new_vertices_list)
    print("\n New Half Edge List \n",new_half_edge_list)
   
   
   
   # Define the filename for the output OBJ file
    output_obj_filename = "output.obj"
   
    # write_obj_file(new_vertices_list, new_face_list, output_obj_filename)
   
   
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