import argparse
from examples.basic_3d_reconstruction import basic_3d_reconstruction_demo
from examples.advanced_scene_understanding import advanced_scene_understanding_demo

def main():
    parser = argparse.ArgumentParser(description="Geometric Deep Learning for 3D Scene Understanding")
    parser.add_argument('--mode', type=str, choices=['demo', 'train', 'process'], default='demo')
    parser.add_argument('--input', type=str, help='Input point cloud or mesh file')
    parser.add_argument('--task', type=str, help='Processing task to perform')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        print("Running Geometric Deep Learning 3D Demo")
        basic_3d_reconstruction_demo()
    
    elif args.mode == 'train':
        print("Running Advanced Training Pipeline")
        advanced_scene_understanding_demo()
    
    elif args.mode == 'process':
        if not args.input:
            print("Please provide an input file with --input")
            return
        
        engine = GeometricEngine()
        
        if args.input.endswith('.ply') or args.input.endswith('.pcd'):
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(args.input)
            pointcloud = np.asarray(pcd.points)
            
            if args.task == "reconstruct":
                mesh = engine.reconstruct_mesh_from_pointcloud(pointcloud)
                o3d.io.write_triangle_mesh(args.output, mesh)
                print(f"Mesh saved to {args.output}")
            
            elif args.task == "understand":
                understanding = engine.understand_scene(pointcloud)
                import json
                with open(args.output, 'w') as f:
                    json.dump(understanding, f, indent=2)
                print(f"Scene understanding results saved to {args.output}")

if __name__ == "__main__":
    main()