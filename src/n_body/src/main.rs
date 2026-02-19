CONST G = 6.67430e-11
CONST THETA = 0.5
CONST MIN_DISTANCE_SQ = 1e-4

STRUCT Vector2D(x: f64, y: f64)
  METHODS:
    zero() -> (0, 0)
    norm_sq() -> x^2 + y^2
    norm() -> sqrt(norm_sq())
    operators: +, -, *, / by scalar

STRUCT Body {
  id: usize
  position: Vector2D
  velocity: Vector2D
  acceleration: Vector2D
  mass: f64
}

STRUCT Rectangle { center_x, center_y, half_width, half_height }
  METHODS:
    contains(point: Vector2D) -> bool
    northwest() -> Rectangle
    northeast() -> Rectangle
    southwest() -> Rectangle
    southeast() -> Rectangle

ENUM QuadTreeNode
  - Empty
  - Leaf(Body)
  - Internal {
      boundary: Rectangle
      children: [QuadTreeNode; 4]     // NW, NE, SW, SE
      center_of_mass: Vector2D
      total_mass: f64
    }

QuadTreeNode.new_internal(boundary) -> Internal with empty children, zero CoM, zero mass

QuadTreeNode.update_mass_properties(self):
  // For Internal node: sum child masses and mass-weighted positions to compute CoM

QuadTreeNode.insert(self, body) -> bool:
  MATCH self:
    Empty:
      replace self with Leaf(body); return true
    Leaf(existing):
      // Parent Internal should convert this quadrant into a new Internal child
      // Create child boundary from quadrant, insert existing then new body
      // Replace child pointer with new Internal; update mass properties; return true
      (handled in Internal case; this path returns false if reached directly)
    Internal { boundary, children, .. }:
      determine quadrant index by body.position relative to boundary center
      compute child_boundary for that quadrant
      MATCH child:
        Empty -> replace with Leaf(body), update mass
        Leaf(existing) -> create new_internal(child_boundary),
                          insert(existing), insert(body),
                          replace child, update mass
        Internal -> recurse insert into child; update mass on success

QuadTreeNode.calculate_force_on_body(self, target_body, theta, min_distance_sq) -> Vector2D:
  MATCH self:
    Empty -> return (0,0)
    Leaf(other_body):
      // Direct interaction if not the same target (or treat as zero force)
      delta = other.position - target.position
      dist_sq = delta.norm_sq()
      if dist_sq < min_distance_sq: return (0,0)
      dist = sqrt(dist_sq)
      force_mag = G * target.mass * other.mass / dist_sq
      return (delta / dist) * force_mag
    Internal { boundary, children, center_of_mass, total_mass }:
      dist_vec = center_of_mass - target.position
      d_sq = dist_vec.norm_sq()
      if d_sq < min_distance_sq: 
        // Avoid singularity; sum children directly
        net = sum(child.calculate_force_on_body(target, theta, min_distance_sq) for child)
        return net
      s = boundary.half_width * 2                   // cell size
      if (s^2)/d_sq < theta^2:                      // opening criterion: s/d < theta
        if total_mass == 0: return (0,0)
        dist = sqrt(d_sq)
        force_mag = G * target.mass * total_mass / d_sq
        return (dist_vec / dist) * force_mag
      else:
        net = sum(child.calculate_force_on_body(target, theta, min_distance_sq) for child)
        return net

CLI (clap):
  -n: number of bodies (default 1000)
  -t: total time (default 1.0)
  -d/--dt: time step (default 0.01)
  --visualize: if true, write CSV "simulation_output.csv"

MAIN:
  parse CLI
  world bounds = Rectangle centered at (0,0) with +/-200 in x/y as half widths
  bodies = []
  for i in 0..n:
    position ~ Uniform([-world_half_width, world_half_width]^2)
    velocity ~ Uniform([-1, 1]^2)
    mass ~ Uniform([1, 1000])
    bodies.push(Body { id, position, velocity, acceleration=0, mass })

  // Build initial Barnes–Hut tree and initial accelerations
  qtree_root = QuadTreeNode.new_internal(root_boundary)
  for body in bodies: qtree_root.insert(body)
  for i in 0..len(bodies):
    target = bodies[i]
    net_force = qtree_root.calculate_force_on_body(target, THETA, MIN_DISTANCE_SQ)
    bodies[i].acceleration = net_force / target.mass (or zero if mass == 0)

  num_steps = ceil(t / dt)
  if visualize: open CSV writer and write header

  for step in 0..num_steps:
    // Rebuild tree for new positions
    current_qtree_root = QuadTreeNode.new_internal(root_boundary)
    for body in bodies: current_qtree_root.insert(body)

    // Update accelerations
    for i in 0..len(bodies):
      target = bodies[i]
      net_force = current_qtree_root.calculate_force_on_body(target, THETA, MIN_DISTANCE_SQ)
      bodies[i].acceleration = net_force / target.mass (or zero)

    // Integrate (Euler–Cromer)
    for body in bodies:
      body.velocity += body.acceleration * dt
      body.position += body.velocity * dt

    // CSV output per step (if visualize)
    if visualize:
      for body in bodies:
        write CSV row: [time_step, id, mass, pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]

  if visualize: flush CSV
  print final states for small n