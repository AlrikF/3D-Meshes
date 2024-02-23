# CSCI599
3D Vision Assignment Template for Spring 2024.

The following tutorial will go through you with how to use three.js for your assignment visualization. Please make sure your VScode is installed with "Live Server" plugin.

## How to use
```shell
git clone https://github.com/jingyangcarl/CSCI599.git
cd CSCI599
ls ./ # you should see index.html and README.md showup in the terminal
code ./ # open this folder via vscode locally
# open and right click on the index.html
# select "Open With Live Server" to run the code over localhost.
```

Half Edge Data Structure Implementation ::

This mainly consists of three components 

Vertices list consist of a dictionary of nodes identified by a vertice name 
Vertice_Edge_Node consists of :
unique name to identify each vertice 
the coordinates of a point in [x, y ,z] format
A half edge originating from the current vertice as its origin 

Face List Consists of a dictionary of all the faces identified by a unique face name 
Face Node consists of :
Face name to uniquely identify the face 
Name of any one of the three half edges the make up the triangle that contains the face 
Normal of the face 

Half edge list consist of a dictionary of edges identified by a unique edge name 
Each Half Edge Node contains:
Name identifying a unique half edge 
The point at which the half edge originates since it is directed 
The point at which the half edge ends 
The twin of the half edge having the same points but directed in the opposite direction 
The name of the face that contains the half edge 
Name of the next half edge of the face 
Name of the prev half edge of the face 



Algorithm of half edge creation :: 
Create the vertices for the vertices list from the trimesh vertices  
Read the triangles from trimesh and create the half edges and faces using the points in the mesh triangles 
Connect the half edges to the twin by checking the origin and end vertices 

Once we create Faces list, Vertices list and Half edge List 

Algorithm for subdivision loop ::
Repeat the following steps for the number of iterations 
    Intialize new_halfedge_list , new_vertices_list, new_face_list

    Iterate through all the edges and compute the odd vertice on each edge 
    Iterate through the vertices and compute the even vertices 
    Adjust the original vertices to the new even vertices and construct new faces and half edges by joining all the odd vertices in a face together. This will result in each face resulting in four more faces 
    Discard the previous halfedges and compute and connect the new twins in the new_halfedge list 
    Discard the old faces and copy the new_vertices to the old vertices_list

Plot and check the final mesh 


Algorithm for decimation :: 
Calculate the normals for each of the faces 
Intialize a while loop where if curr no of faces is is greater than required faces we enter the loop 
    Calculate the error for each point by computing K matrix and summing it up for all faces to get Q matrix 
    Iterate through all the edges and compute the new vertex if we collapse the edge and the corresponding error and push it into a heap 
    Pop the edge with the least error delete the vertices, half edges and faces associated with the points and join the half edges of the remaining faces to get the correct twin half edge 

Plot and check the final decimated mesh 
















