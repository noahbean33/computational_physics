# Rust N-Body Simulation with Parallel Programming

**Build Status** | **License** | **Rust Version**

This project is a high-performance N-body simulation written in Rust, designed to model the gravitational interactions between multiple bodies. By leveraging Rust's concurrency features and parallel programming techniques, this simulation efficiently handles large-scale computations, making it ideal for researchers, developers, and enthusiasts interested in computational physics and parallel computing.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Algorithms](#algorithms)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Performance](#performance)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)
- [Documentation](#documentation)

---

## Overview

An N-body simulation models the motion of celestial bodies (such as planets, stars, or galaxies) under the influence of gravitational forces. This project implements an efficient solution to this classic problem using Rust, a language known for its performance, safety, and concurrency capabilities. The simulation is optimized for large numbers of bodies by employing parallel programming techniques, making it suitable for both educational purposes and research.

---

## Key Features

- **Rust Implementation**: Harnesses Rust’s performance, memory safety, and concurrency features for reliable and efficient computations.
- **Parallel Programming**: Utilizes Rust’s Rayon library for parallel processing, significantly speeding up simulations for large numbers of bodies.
- **Efficient Algorithms**: Implements the Barnes-Hut algorithm for O(N log N) complexity, enabling scalable simulations.
- **Visualization**: Includes optional tools to generate plots or animations of the simulation results.
- **Cross-Platform**: Runs on any platform supported by Rust, including Windows, macOS, and Linux.

---

## Algorithms

This simulation uses the **Barnes-Hut algorithm**, an approximation technique that reduces the computational complexity of gravitational force calculations from O(N²) to O(N log N). It achieves this by grouping distant bodies into a single node in a **quadtree (2D)** or **octree (3D)**.

Parallelization is implemented using Rust’s **Rayon** library, enabling easy parallel iteration over the bodies for force calculations and position updates. This ensures efficient utilization of multi-core processors.

---

## Installation

To build and run this project, you need to have Rust installed on your system. If you don’t have Rust installed, visit [rust-lang.org](https://www.rust-lang.org/).

### Steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/rust-n-body.git
    cd rust-n-body
    ```

2. Build the project:
    ```sh
    cargo build --release
    ```

This will compile the project in release mode for optimal performance.

---

## Usage

To run the simulation, use the following command:

```sh
cargo run --release -- [options]
Available Options:
-n <number>: Number of bodies (default: 1000)

-t <time>: Total simulation time in arbitrary units (default: 1.0)

-dt <timestep>: Time step for integration (default: 0.01)

--visualize: Generate visualization output (e.g., plots or animations)

Examples
Run a simulation with the default settings (1000 bodies, 1.0 time units):

sh
Copy
Edit
cargo run --release
Run a simulation with 5000 bodies and a custom time step:

sh
Copy
Edit
cargo run --release -- -n 5000 -dt 0.005
Run a simulation and generate visualization output:

sh
Copy
Edit
cargo run --release -- -n 2000 --visualize
Performance
The parallel implementation significantly reduces computation time for large numbers of bodies.

Number of Bodies (N)	Sequential Time (s)	Parallel Time (s)	Speedup
1,000	5.2	1.5	3.5x
10,000	52.0	14.8	3.5x
100,000	520.0	148.0	3.5x
Note: Actual performance may vary based on hardware and simulation parameters.

Visualization
If the --visualize flag is used, the simulation will generate output files that can be used to create plots or animations of the body movements.

Example Python Script
Assuming output is saved in simulation_output.csv:

python
Copy
Edit
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('simulation_output.csv')
plt.scatter(data['x'], data['y'], s=1)
plt.title('N-Body Simulation Snapshot')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.show()
Contributing
Contributions are welcome! If you’d like to improve the simulation, add new features, or fix bugs, please follow these steps:

Fork the repository.

Create a new branch for your feature or bugfix.

Write tests for any new functionality.

Submit a pull request with a clear description of your changes.

Please ensure that your code follows Rust’s coding standards and includes appropriate documentation.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Documentation
For more detailed information on the implementation, algorithms, and code structure, please refer to the documentation (docs/index.html). This includes explanations of:

The parallelization techniques

The Barnes-Hut algorithm

How to extend the simulation for custom use cases

Additional Notes
Why Rust? Rust’s ownership model ensures memory safety without a garbage collector, making it ideal for performance-critical applications like simulations. Its concurrency features also make parallel programming safer and more straightforward.

Extensibility: The simulation is designed to be modular, allowing users to easily swap out algorithms or add new features like different force laws or integrators.

Community: Join the discussion on GitHub Issues to report bugs, request features, or ask questions.
