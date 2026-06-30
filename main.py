import argparse
from visualize import plot_results

def main():
    parser = argparse.ArgumentParser(description="Run inference on a TESS target using the Tri-Input model.")
    parser.add_argument("--target", type=str, required=True, help="TIC ID (e.g., 'TIC 123495874') or path to FITS file.")
    parser.add_argument("--model", type=str, default="best_model.keras", help="Path to the trained keras model.")
    
    args = parser.parse_args()
    
    plot_results(args.target, args.model)

if __name__ == "__main__":
    main()
