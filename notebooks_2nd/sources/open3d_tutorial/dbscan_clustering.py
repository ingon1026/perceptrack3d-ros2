# Open3D tutorial: Point cloud — https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
# Source notebook: https://github.com/isl-org/Open3D/blob/main/docs/jupyter/geometry/pointcloud.ipynb (MIT)
# Fetched 2026-09-08. Code cell copied verbatim; the markdown above it is quoted as a comment.

# --- markdown ---
# ## DBSCAN clustering
# Given a point cloud from e.g. a depth sensor we want to group local point cloud clusters together. For this purpose, we can use clustering algorithms. Open3D implements DBSCAN [\[Ester1996\]](../reference.html#Ester1996) that is a density based clustering algorithm. The algorithm is implemented in `cluster_dbscan` and requires two parameters: `eps` defines the distance to neighbors in a cluster and `min_points` defines the minimum number of points required to form a cluster. The function returns `labels`, where the label `-1` indicates noise.

# --- code cell ---
ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud(ply_point_cloud.path)

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.455,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])
