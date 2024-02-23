import trimesh
import matplotlib.pyplot as plt
from heapq import heapify, heappush, heappop
import numpy as np
import copy

# Each vertex has its own unique Vertice Edge node 
class Vertice_Edge_Node:
    def __init__(self,name, coordinates, incident_edge=None):
        self.vname= name
        self.coordinates= coordinates
        self.incident_edge= incident_edge
        self.Q_error=None

    
    def __str__(self):  
        return f"\n vertice : {self.vname},  coordinates: {self.coordinates}, incident: {self.incident_edge} , Q: {self.Q_error}|\n"
    
    def __repr__(self):  
        return f"\n vertice : {self.vname},  coordinates: {self.coordinates}, incident: {self.incident_edge} , Q: {self.Q_error} |\n"


# Each Face has a unique Face node identified by a  unique name 
class Face_Node:
    def __init__(self,name, half_edge):
        self.fname = name
        self.half_edge= half_edge
        self.normal= None

    def __str__(self):  
        return f"\n face : {self.fname},  half edge: {self.half_edge} , Normal: {self.normal}|\n"
    
    def __repr__(self):  
        return f"\n face : {self.fname},  half edge: {self.half_edge} , Normal: {self.normal}|\n"

    

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
        
        # print("Triangle:",triangle)
        
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
            vertices_list[v_name].incident_edge = f"e{edge_count}"
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
        
        
        print(f"Org: {org}, End {end}, {org_coords}, {end_coords}")
        ax.plot([org_coords[0], end_coords[0]], [org_coords[1], end_coords[1]], [org_coords[2], end_coords[2]], color='b')

    # Set labels and show plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
          

    # Annotating vertices
    for vertex_name in vertices_list:
        ax.scatter(vertices_list[vertex_name].coordinates[0], vertices_list[vertex_name].coordinates[1], vertices_list[vertex_name].coordinates[2], color='g')
        ax.text(vertices_list[vertex_name].coordinates[0], vertices_list[vertex_name].coordinates[1], vertices_list[vertex_name].coordinates[2], vertex_name)

        
    for vertex_name in new_vertices_list:
        ax.scatter(new_vertices_list[vertex_name].coordinates[0], new_vertices_list[vertex_name].coordinates[1], new_vertices_list[vertex_name].coordinates[2], color='r')
        ax.text(new_vertices_list[vertex_name].coordinates[0], new_vertices_list[vertex_name].coordinates[1], new_vertices_list[vertex_name].coordinates[2], vertex_name)
        

    plt.show()

def compute_odd_vertices(mesh,half_edge_list,vertices_list,new_vertices_list, debug=False):
    visited=set([])
    for e_name in half_edge_list:
        if e_name not in visited:
            if debug:
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




def compute_even_vertices(mesh,half_edge_list,vertices_list, new_vertices_list, debug=False):
    
    for vertex_name in vertices_list:
        neighbours = set([])
        for he_name in half_edge_list:
            if half_edge_list[he_name].origin == vertex_name:
                neighbours.add(half_edge_list[he_name].end_vertex)
        sum_neighbours = 0
        for neighbour in neighbours:
            sum_neighbours+= vertices_list[neighbour].coordinates
        if debug :
            print("Sum neighbours :", sum_neighbours)

        beta=3/16.0
        if len(neighbours)>3:
            beta=1/len(neighbours)*(5/8.0 - (3/8.0 + 1/4.0 * np.cos(2*np.pi/len(neighbours)))**2)
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

    




def compute_new_faces(faces_list, half_edge_list, new_vertices_list,new_half_edge_list,new_face_list, debug=False):
    
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
        child_odd_vertex3 = half_edge_list[next_he_name].child_odd_vertex

        if debug:
            print("Creating 4 new faces")
            print([origin_vertex1,child_odd_vertex1,child_odd_vertex2])
            print([origin_vertex2,child_odd_vertex2,child_odd_vertex3])
            print([origin_vertex3,child_odd_vertex3,child_odd_vertex1])
            print([child_odd_vertex1,child_odd_vertex3,child_odd_vertex2])

        create_new_faces(new_face_list,new_half_edge_list,[origin_vertex1,child_odd_vertex1,child_odd_vertex2])
        create_new_faces(new_face_list,new_half_edge_list,[origin_vertex2,child_odd_vertex2,child_odd_vertex3])
        create_new_faces(new_face_list,new_half_edge_list,[origin_vertex3,child_odd_vertex3,child_odd_vertex1])
        create_new_faces(new_face_list,new_half_edge_list,[child_odd_vertex1,child_odd_vertex3,child_odd_vertex2])
        # print(new_face_list)

def connect_twin_half_edges(new_half_edge_list):
    for he_name in new_half_edge_list:
        if new_half_edge_list[he_name].twin_name is None:
            for twin_he_name in new_half_edge_list:
                if (new_half_edge_list[twin_he_name].origin == new_half_edge_list[he_name].end_vertex and
                    new_half_edge_list[twin_he_name].end_vertex == new_half_edge_list[he_name].origin):
                    new_half_edge_list[he_name].twin_name = twin_he_name
                    new_half_edge_list[twin_he_name].twin_name = he_name 
                    break

def subdivision_loop(mesh, vertices_list, half_edge_list, faces_list, iterations=1, debug =True):
    """
    Apply Loop subdivision to the input mesh for the specified number of iterations.
    :param mesh: input mesh
    :param iterations: number of iterations
    :return: mesh after subdivision
    """
    new_vertices_list = copy.deepcopy(vertices_list)
    new_half_edge_list = {}
    new_face_list = {}
    for iter in range(iterations):
        new_half_edge_list = {}
        new_face_list = {}
        # new_half_edge_list= copy.deepcopy(half_edge_list)
        new_vertices_list = copy.deepcopy(vertices_list)
        print(f"Iteration {iter}")
        print(half_edge_list)


        compute_odd_vertices(mesh,half_edge_list=half_edge_list,vertices_list=vertices_list,new_vertices_list=new_vertices_list )
      
        compute_even_vertices(mesh,half_edge_list=half_edge_list,vertices_list=vertices_list,new_vertices_list=new_vertices_list)
        
        compute_new_faces(faces_list,half_edge_list, new_vertices_list,new_half_edge_list,new_face_list )
        
        connect_twin_half_edges(new_half_edge_list=new_half_edge_list)
                
        
        half_edge_list= copy.deepcopy(new_half_edge_list)
        faces_list = copy.deepcopy(new_face_list)
        vertices_list = copy.deepcopy(new_vertices_list)

        print(f"End of iteration {iter}")
        print("\n Half Edge List ::\n", half_edge_list)

        print(" \n Faces List : \n ", faces_list)

        print("\n Vertices List \n",vertices_list)



    print(len(new_half_edge_list),len(new_vertices_list),len(new_face_list)) 
    plot_half_edge(new_half_edge_list,new_vertices_list)
    

    
    return new_half_edge_list, new_vertices_list,new_face_list



###################################################   DECIMATION    ##############################################################

def calculate_normal(face_name,face_list,half_edge_list,vertices_list):
    e1 = face_list[face_name].half_edge
    v1 = half_edge_list[e1].origin
    v2 = half_edge_list[e1].end_vertex
    v3 = half_edge_list[half_edge_list[e1].next].end_vertex

    v1_v2_vector = vertices_list[v2].coordinates - vertices_list[v1].coordinates
    v1_v3_vector = vertices_list[v3].coordinates - vertices_list[v1].coordinates

    normal  = np.cross(v1_v2_vector, v1_v3_vector)
    print("Normal",normal)
    magnitude = np.linalg.norm(normal)  # Calculate the magnitude of the vector
    if magnitude != 0:
        normal = normal/magnitude  # If the vector is the zero vector, return the same vector

    d =  np.dot(normal, vertices_list[v1].coordinates)
    face_list[face_name].normal = np.append(normal, -d) 


def calculate_point_error(vertex_name,face_list,half_edge_list,vertices_list, debug =True):
    curr_edge=vertices_list[vertex_name].incident_edge
    print("Curr edge",curr_edge)
    visited_faces=set([])
    if debug:
        print(f"Calculating Error for point : { vertex_name} ,{curr_edge}")
    face = half_edge_list[curr_edge].incident_face
    
    Q=np.zeros((4,4))
    
    while(face not in visited_faces):
        visited_faces.add(face)
        p = face_list[face].normal
        p = np.expand_dims(p, axis=1)
        # print("K:",type(p), p, np.dot(p, p.T) )
        Q+= np.dot(p, p.T)    
        print(face_list[ face])

        curr_edge = half_edge_list[half_edge_list[curr_edge].prev].twin_name
        face = half_edge_list[curr_edge].incident_face

    matrix_v = np.append(vertices_list[vertex_name].coordinates,1)
    print("Matrix V",matrix_v)

    vertices_list[vertex_name].Q_error = Q

def least_pair_error(half_edge_list, vertices_list):

    error_heap = [] 
    heapify(error_heap)
    for edge_name in half_edge_list:
        v1 = half_edge_list[edge_name].origin
        v2 = half_edge_list[edge_name].end_vertex

        Q_bar = vertices_list[v1].Q_error +vertices_list[v2].Q_error
        der_q = Q_bar.copy()
        der_q[-1] = [0, 0, 0, 1]
        # print("Q_bar",der_q)
        determinant = np.linalg.det(der_q)
        # If the determinant is not zero, calculate the optimal position of the vertex as matrix is invertible
        if determinant != 0:
            
            Q_bar_inverse = np.linalg.inv(der_q)
            column_vector = np.array([[0],[0],[0],[1]])
            v_bar = np.dot(Q_bar_inverse, column_vector)
            v_bar = v_bar.flatten() 
            print("\n  INVERTIBLE \n\n ",v_bar)
        else:
            ### Check if only midpoint is the optimal position
            print("\n  NOT INVERTIBLE \n\n ")
            min_error =np.inf
            v_mid = (vertices_list[v1].coordinates + vertices_list[v2].coordinates)/2
            v_bar=v_mid
            
            for coord in [vertices_list[v1].coordinates,vertices_list[v2].coordinates,v_mid] :
                coord=np.append(coord,1)
                error = np.dot(np.dot(coord.T, Q_bar), coord)
                if error<min_error:
                    min_error =error
                    v_bar=coord
            
      
        pair_error = np.dot(np.dot(v_bar.T, Q_bar), v_bar)
        
        heappush(error_heap, [pair_error, edge_name, v_bar[:-1]/v_bar[-1]])

    print("Error Heap",error_heap)
    # Pop the edge with the smallest error
    pair_error, edge_name, v_bar = heappop(error_heap)
    # print("\n \n \n \n Pair Error",pair_error, edge_name, v_bar)
    return pair_error, edge_name, v_bar

def collapse_edge(edge_name, v_bar, half_edge_list, vertices_list, faces_list, debug =True):
    # Get the vertices of the edge
    v1 = half_edge_list[edge_name].origin
    v2 = half_edge_list[edge_name].end_vertex
    print(type(vertices_list.keys()))
    last_key = int((list(vertices_list.keys())[-1])[1:]) + 1

    new_node_name = f"v{last_key}"
    # Update the coordinates of v1 to be the optimal position
    print("Vbar: ", v_bar)
    vertices_list[new_node_name] = Vertice_Edge_Node(new_node_name, v_bar) 

    del_edge1 = edge_name
    del_edge2 = half_edge_list[edge_name].next
    del_edge3 = half_edge_list[edge_name].prev
    twin_2 = half_edge_list[del_edge2].twin_name
    twin_3 = half_edge_list[del_edge3].twin_name
    del_face1 = half_edge_list[del_edge1].incident_face
    if vertices_list[half_edge_list[del_edge3].origin].incident_edge== del_edge3:
        vertices_list[half_edge_list[del_edge3].origin].incident_edge = twin_2
    

    del_edge4 = half_edge_list[edge_name].twin_name
    del_edge5 = half_edge_list[del_edge4].next
    del_edge6 = half_edge_list[del_edge4].prev
    twin5 = half_edge_list[del_edge5].twin_name
    twin6 = half_edge_list[del_edge6].twin_name
    del_face2 = half_edge_list[del_edge4].incident_face
    
    if vertices_list[half_edge_list[del_edge6].origin].incident_edge== del_edge6:
        vertices_list[half_edge_list[del_edge6].origin].incident_edge = twin5
    
    

    if debug:
        print(f"Edges to be deleted are ::",[del_edge1,del_edge2,del_edge3,del_edge4,del_edge5,del_edge6])


    # Update twin pointers
    half_edge_list[twin_2].twin_name = twin_3
    half_edge_list[twin_3].twin_name = twin_2

    half_edge_list[twin5].twin_name = twin6
    half_edge_list[twin6].twin_name = twin5

    if half_edge_list[twin_2].origin == v2 or half_edge_list[twin_2].origin == v1:
        start_edge = twin_2
    else:
        start_edge = twin_3

    vertices_list[new_node_name].incident_edge=start_edge
    
    curr_edge = start_edge
    start_edge=  None
    while(curr_edge != start_edge):
        if start_edge==None:
            start_edge=curr_edge
        half_edge_list[curr_edge].origin = new_node_name
        prev_edge = half_edge_list[curr_edge].prev
        half_edge_list[prev_edge].end_vertex = new_node_name
        if debug:
            print(f"Changing vertex for {curr_edge, prev_edge}")
            print("iterating ")

        curr_edge= half_edge_list[prev_edge].twin_name

    for edge in [del_edge1, del_edge2, del_edge3, del_edge4, del_edge5, del_edge6]:
        del half_edge_list[edge]
        
    del faces_list[del_face1]
    del faces_list[del_face2]

    del vertices_list[v1]
    del vertices_list[v2]

    calculate_normal(half_edge_list[twin_2].incident_face,faces_list,half_edge_list,vertices_list)
    calculate_normal(half_edge_list[twin_3].incident_face,faces_list,half_edge_list,vertices_list)
    calculate_normal(half_edge_list[twin5].incident_face,faces_list,half_edge_list,vertices_list)
    calculate_normal(half_edge_list[twin6].incident_face,faces_list,half_edge_list,vertices_list)



def simplify_quadric_error(mesh,vertices_list,half_edge_list, faces_list,  face_count=1):
    """
    Apply quadratic error mesh decimation to the input mesh until the target face count is reached.
    :param mesh: input mesh
    :param face_count: number of faces desired in the resulting mesh.
    :return: mesh after decimation
    """
    print("Vertex list:: ")
    print(vertices_list)

    for face_name in faces_list:
        calculate_normal(face_name,faces_list,half_edge_list,vertices_list)

    

    while face_count < len(faces_list):
        print("half edge list:: ", half_edge_list)

        for vertex_name in vertices_list:
            calculate_point_error(vertex_name,faces_list,half_edge_list,vertices_list)

        print("Face list:: " , faces_list)

        print("Vertices list ",vertices_list)

        pair_error, edge_name, v_bar =  least_pair_error(half_edge_list, vertices_list )
        print(f" \n Removing edge : {half_edge_list[edge_name]} "  )
        collapse_edge(edge_name, v_bar, half_edge_list, vertices_list, faces_list)

        print(half_edge_list)
        print(vertices_list)

    plot_half_edge(half_edge_list,vertices_list)


    return 



def create_trimesh(v_list, f_list, he_list):
    vertices = []
    faces = []
    # Convert vertices
    cnt=0
    mp={}
    for vertex in v_list:
        vertices.append(v_list[vertex].coordinates)
        mp[v_list[vertex].vname[1:]]=cnt
        cnt+=1

    # Convert faces
    for fc in f_list:
        half_edge = f_list[fc].half_edge
        v1, v2, v3 = he_list[half_edge].origin, he_list[half_edge].end_vertex, he_list[he_list[half_edge].next].end_vertex
        faces.append([mp[v1[1:]], mp[v2[1:]], mp[v3[1:]] ])

    # Convert lists to numpy arrays
    vertices = np.array(vertices)
    faces = np.array(faces)

    # Create trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    return mesh





if __name__ == '__main__':
    # Load mesh and print information
    # mesh = trimesh.load_mesh('assets/cube.obj')
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    print(f'Mesh Info: {mesh}')
    
    # apply loop subdivision over the loaded mesh
    # mesh_subdivided = mesh.subdivide_loop(iterations=1)

    
    print("Vertices :", mesh.vertices)

    vertices_list = {}
    

    half_edge_list={}

    faces_list = {}

    create_half_edge(mesh=mesh,vertices_list=vertices_list, half_edge_list=half_edge_list, faces_list=faces_list)

    # print("\n Half Edge List ::\n", half_edge_list)

    # print(" \n Faces List : \n ", faces_list)

    # print("\n Vertices List \n",vertices_list)

    

    # # TODO: implement your own loop subdivision here
    new_half_edge_list, new_vertices_list, new_face_list = subdivision_loop(mesh, vertices_list, half_edge_list, faces_list, iterations=3)
    
   
    

    uniq_edges = set([])
    for e in new_half_edge_list:
        if f"{new_half_edge_list[e].origin}_{new_half_edge_list[e].end_vertex}" in uniq_edges:
            print(f"{new_half_edge_list[e].origin}_{new_half_edge_list[e].end_vertex}" + " is repeated ")
        uniq_edges.add(f"{new_half_edge_list[e].origin}_{new_half_edge_list[e].end_vertex}")

    print(len(uniq_edges))
    print(len(new_vertices_list),len(new_half_edge_list), len(new_face_list))
   
    # Define the filename for the output OBJ file
    output_obj_filename = "output.obj"
   
    # Call the function to create a Trimesh object
    print(mesh.faces)

    mesh_subdivided = create_trimesh(v_list=new_vertices_list, f_list=new_face_list, he_list=new_half_edge_list )
    
    
   
    #print the new mesh information and save the mesh
    print(f'Subdivided Mesh Info: {mesh_subdivided}')
    mesh_subdivided.export('assets/assignment1/cube_subdivided.obj')



    
    vertices_list = {}
    
    half_edge_list={}

    faces_list = {}

    create_half_edge(mesh=mesh,vertices_list=vertices_list, half_edge_list=half_edge_list, faces_list=faces_list)

    # print("\n Half Edge List ::\n", half_edge_list)

    # print(" \n Faces List : \n ", faces_list)

    # print("\n Vertices List \n",vertices_list)
    
    # quadratic error mesh decimation
    # actual_mesh_decimated = mesh.simplify_quadric_decimation(8)
    
    # # TODO: implement your own quadratic error mesh decimation here
    simplify_quadric_error(mesh, vertices_list, half_edge_list, faces_list, face_count=10)

    mesh_decimated = create_trimesh(v_list=vertices_list, f_list=faces_list, he_list=half_edge_list )
    print(faces_list)
    # # print the new mesh information and save the mesh
    print(f'Decimated Mesh Info: {mesh_decimated}')
    mesh_decimated.export('assets/assignment1/cube_decimated.obj')
    # actual_mesh_decimated.show()
    mesh_decimated.show()
    mesh_decimated.export('cube_decimated.obj')