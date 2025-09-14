use std::f32::consts::{FRAC_PI_2, PI};

use bevy::{
    input::mouse::MouseMotion,
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
};
use bevy_inspector_egui::{bevy_egui::EguiPlugin, prelude::*, quick::WorldInspectorPlugin};
use noise::Perlin;

const CUBOID_WIDTH: f32 = 2.0;
const HALF_CUBOID_WIDTH: f32 = CUBOID_WIDTH / 2.0;
const CUBOID_DEPTH: f32 = 0.2;

const MOVEMENT_SPEED: f32 = 5.0;
const DRAG_SENSITIVITY: f32 = 0.004;

#[derive(Resource, Reflect, Default, InspectorOptions)]
#[reflect(Resource, InspectorOptions)]
struct MeshSettings {
    #[inspector(min = 2, max = 200)]
    resolution: usize,
    #[inspector(min = 0.0, max = 10.0)]
    noise_scale: f64,
    #[inspector(min = 0.0, max = 2.0)]
    noise_height: f32,
}

#[derive(Component)]
struct PerlinMesh;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            EguiPlugin::default(),
            WorldInspectorPlugin::default(),
        ))
        .register_type::<MeshSettings>()
        .insert_resource(MeshSettings {
            resolution: 60,
            noise_scale: 1.0,
            noise_height: 0.3,
        })
        .add_systems(Startup, setup)
        .add_systems(Update, (fly_camera, drag_camera, regenerate_on_spacebar))
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    settings: Res<MeshSettings>,
) {
    // Camera
    commands.spawn((
        Name::new("Main Camera"),
        Camera3d::default(),
        Transform::from_xyz(0.0, 3.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Light
    commands.spawn((
        Name::new("Main Light"),
        PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));

    // Origin
    commands.spawn((
        Name::new("Origin"),
        Mesh3d(meshes.add(Cuboid::new(0.1, 0.1, 0.1))),
        MeshMaterial3d(materials.add(Color::srgb(0.7, 0.3, 0.3))),
        Transform::default(),
    ));

    spawn_perlin_meshes(&mut commands, &mut meshes, &mut materials, &settings);
}

fn fly_camera(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut query: Query<&mut Transform, With<Camera3d>>,
) {
    let mut direction = Vec3::ZERO;
    if keyboard.pressed(KeyCode::KeyD) {
        direction.x += 1.0;
    }
    if keyboard.pressed(KeyCode::KeyA) {
        direction.x -= 1.0;
    }
    if keyboard.pressed(KeyCode::KeyQ) {
        direction.y += 1.0;
    }
    if keyboard.pressed(KeyCode::KeyX) {
        direction.y -= 1.0;
    }
    if keyboard.pressed(KeyCode::KeyW) {
        direction.z += 1.0;
    }
    if keyboard.pressed(KeyCode::KeyS) {
        direction.z -= 1.0;
    }

    if direction != Vec3::ZERO {
        direction = direction.normalize();
        for mut transform in &mut query {
            // Move relative to camera's local axes
            let forward = transform.forward();
            let right = transform.right();
            let up = Vec3::Y;
            let movement = forward * direction.z + right * direction.x + up * direction.y;
            transform.translation += movement * MOVEMENT_SPEED * time.delta_secs();
        }
    }
}

fn drag_camera(
    mouse_button: Res<ButtonInput<MouseButton>>,
    mut mouse_motion_events: EventReader<MouseMotion>,
    mut query: Query<&mut Transform, With<Camera3d>>,
) {
    if mouse_button.pressed(MouseButton::Left) {
        let mut delta = Vec2::ZERO;
        for event in mouse_motion_events.read() {
            delta += event.delta;
        }
        if delta != Vec2::ZERO {
            for mut transform in &mut query {
                // Yaw (around global Y)
                let yaw = Quat::from_rotation_y(-delta.x * DRAG_SENSITIVITY);
                // Pitch (around local X)
                let pitch = Quat::from_rotation_x(-delta.y * DRAG_SENSITIVITY);
                transform.rotation = yaw * transform.rotation; // yaw first
                transform.rotation = transform.rotation * pitch; // then pitch
            }
        }
    }
}

fn regenerate_on_spacebar(
    mut commands: Commands,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    settings: Res<MeshSettings>,
    query: Query<Entity, With<PerlinMesh>>,
) {
    if keyboard_input.just_pressed(KeyCode::Space) {
        // Despawn old meshes
        for entity in &query {
            commands.entity(entity).despawn();
        }

        // Spawn new ones with updated settings
        spawn_perlin_meshes(&mut commands, &mut meshes, &mut materials, &settings);
    }
}

fn spawn_perlin_meshes(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    settings: &MeshSettings,
) {
    let perlin = Perlin::new(12345);

    let mesh_000 = generate_cell_mesh(Vec3::new(0.0, 0.0, 0.0), &perlin, settings);
    let mesh_001 = generate_cell_mesh(Vec3::new(0.0, 0.0, CUBOID_WIDTH), &perlin, settings);
    let mesh_100 = generate_cell_mesh(Vec3::new(CUBOID_WIDTH * 2.0, 0.0, 0.0), &perlin, settings);

    commands.spawn((
        Name::new("Mesh 000"),
        PerlinMesh,
        Mesh3d(meshes.add(mesh_000)),
        MeshMaterial3d(materials.add(Color::srgb(0.6, 0.6, 1.0))),
        Transform::default(),
    ));

    // commands.spawn((
    //     Name::new("Mesh 001"),
    //     PerlinMesh,
    //     Mesh3d(meshes.add(mesh_001)),
    //     MeshMaterial3d(materials.add(Color::srgb(0.3, 0.7, 0.9))),
    //     Transform::from_xyz(0.0, 0.0, CUBOID_WIDTH),
    // ));

    // commands.spawn((
    //     Name::new("Mesh 100"),
    //     PerlinMesh,
    //     Mesh3d(meshes.add(mesh_100)),
    //     MeshMaterial3d(materials.add(Color::srgb(0.3, 0.7, 0.9))),
    //     Transform::from_xyz(CUBOID_WIDTH, 0.0, 0.0),
    // ));
}

fn generate_cell_mesh(origin: Vec3, perlin: &Perlin, settings: &MeshSettings) -> Mesh {
    let mesh_x_pos = generate_wall_mesh(&perlin, settings, Direction::XPos);
    let mesh_x_neg = generate_wall_mesh(&perlin, settings, Direction::XNeg);
    let mesh_y_pos = generate_wall_mesh(&perlin, settings, Direction::YPos);
    let mesh_y_neg = generate_wall_mesh(&perlin, settings, Direction::YNeg);
    let mesh_z_pos = generate_wall_mesh(&perlin, settings, Direction::ZPos);
    let mesh_z_neg = generate_wall_mesh(&perlin, settings, Direction::ZNeg);
    merge_meshes!(mesh_x_pos, mesh_x_neg, mesh_y_pos, mesh_y_neg, mesh_z_pos, mesh_z_neg)
}

#[derive(Debug, Clone, Reflect)]
pub enum Direction {
    XPos,
    XNeg,
    YPos,
    YNeg,
    ZPos,
    ZNeg,
}

impl Direction {
    fn mesh_translation(&self) -> Vec3 {
        match self {
            Self::XPos => Vec3::new(-HALF_CUBOID_WIDTH, HALF_CUBOID_WIDTH, -HALF_CUBOID_WIDTH),
            Self::XNeg => Vec3::new(HALF_CUBOID_WIDTH, -HALF_CUBOID_WIDTH, -HALF_CUBOID_WIDTH),
            Self::YPos => Vec3::new(HALF_CUBOID_WIDTH, HALF_CUBOID_WIDTH, -HALF_CUBOID_WIDTH),
            Self::YNeg => Vec3::new(-HALF_CUBOID_WIDTH, -HALF_CUBOID_WIDTH, -HALF_CUBOID_WIDTH),
            Self::ZPos => Vec3::new(-HALF_CUBOID_WIDTH, -HALF_CUBOID_WIDTH, HALF_CUBOID_WIDTH),
            Self::ZNeg => Vec3::new(-HALF_CUBOID_WIDTH, HALF_CUBOID_WIDTH, -HALF_CUBOID_WIDTH),
        }
    }

    fn mesh_rotation(&self) -> Quat {
        match self {
            Self::XPos => Quat::from_rotation_z(-FRAC_PI_2),
            Self::XNeg => Quat::from_rotation_z(FRAC_PI_2),
            Self::YPos => Quat::from_rotation_z(PI),
            Self::YNeg => Quat::default(),
            Self::ZPos => Quat::from_rotation_x(-FRAC_PI_2),
            Self::ZNeg => Quat::from_rotation_x(FRAC_PI_2),
        }
    }
}

fn generate_wall_mesh(perlin: &Perlin, settings: &MeshSettings, direction: Direction) -> Mesh {
    let origin = Vec3::ZERO;

    // 8 cuboid corners
    let top_y = CUBOID_DEPTH;
    let bottom_y = 0.0;
    let corners = [
        // bottom
        [origin.x, bottom_y, origin.z],                // 0
        [origin.x + CUBOID_WIDTH, bottom_y, origin.z], // 1
        [origin.x + CUBOID_WIDTH, bottom_y, origin.z + CUBOID_WIDTH], // 2
        [origin.x, bottom_y, origin.z + CUBOID_WIDTH], // 3
        // top
        [origin.x, top_y, origin.z],                               // 4
        [origin.x + CUBOID_WIDTH, top_y, origin.z],                // 5
        [origin.x + CUBOID_WIDTH, top_y, origin.z + CUBOID_WIDTH], // 6
        [origin.x, top_y, origin.z + CUBOID_WIDTH],                // 7
    ];

    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    for (i, &corner) in corners.iter().enumerate() {
        positions.push(corner);
        let normal = match i {
            0 | 1 | 2 | 3 => [0.0, -1.0, 0.0], // bottom
            4 | 5 | 6 | 7 => [0.0, 1.0, 0.0],  // top
            _ => [0.0, 0.0, 0.0],
        };
        normals.push(normal);
        let uv = match i {
            0 => [0.0, 0.0],
            1 => [1.0, 0.0],
            2 => [1.0, 1.0],
            3 => [0.0, 1.0],
            4 => [0.0, 0.0],
            5 => [1.0, 0.0],
            6 => [1.0, 1.0],
            7 => [0.0, 1.0],
            _ => [0.0, 0.0],
        };
        uvs.push(uv);
    }

    // Indices for 12 triangles (2 per face)
    indices.extend([
        // bottom
        0, 1, 2, 0, 2, 3, // top
        4, 6, 5, 4, 7, 6, // +X
        1, 5, 6, 1, 6, 2, // -X
        0, 3, 7, 0, 7, 4, // +Z
        2, 6, 7, 2, 7, 3, // -Z
        0, 4, 5, 0, 5, 1,
    ]);

    // Rotation must be applied before translation, so the axes stay correct
    rotate(&mut positions, &mut normals, direction.mesh_rotation());
    translate(&mut positions, direction.mesh_translation());

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        bevy::render::render_asset::RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));

    mesh
}

fn translate(positions: &mut Vec<[f32; 3]>, offset: Vec3) {
    for pos in positions.iter_mut() {
        pos[0] += offset.x;
        pos[1] += offset.y;
        pos[2] += offset.z;
    }
}

fn rotate(positions: &mut Vec<[f32; 3]>, normals: &mut Vec<[f32; 3]>, rotation: Quat) {
    for pos in positions.iter_mut() {
        let v = Vec3::new(pos[0], pos[1], pos[2]);
        let rotated = rotation * v;
        pos[0] = rotated.x;
        pos[1] = rotated.y;
        pos[2] = rotated.z;
    }
    for normal in normals.iter_mut() {
        let v = Vec3::new(normal[0], normal[1], normal[2]);
        let rotated = rotation * v;
        normal[0] = rotated.x;
        normal[1] = rotated.y;
        normal[2] = rotated.z;
    }
}

fn merge_meshes(mesh_a: &Mesh, mesh_b: &Mesh) -> Mesh {
    // Get attributes from both meshes
    let mut positions: Vec<[f32; 3]> = mesh_a
        .attribute(Mesh::ATTRIBUTE_POSITION)
        .unwrap()
        .as_float3()
        .unwrap()
        .to_vec();
    let mut normals: Vec<[f32; 3]> = mesh_a
        .attribute(Mesh::ATTRIBUTE_NORMAL)
        .unwrap()
        .as_float3()
        .unwrap()
        .to_vec();
    let mut uvs: Vec<[f32; 2]> = match mesh_a.attribute(Mesh::ATTRIBUTE_UV_0).unwrap() {
        VertexAttributeValues::Float32x2(values) => values.clone(),
        _ => panic!("UV_0 attribute is not Float32x2"),
    };

    let positions_b = mesh_b
        .attribute(Mesh::ATTRIBUTE_POSITION)
        .unwrap()
        .as_float3()
        .unwrap();
    let normals_b = mesh_b
        .attribute(Mesh::ATTRIBUTE_NORMAL)
        .unwrap()
        .as_float3()
        .unwrap();
    let uvs_b: Vec<[f32; 2]> = match mesh_a.attribute(Mesh::ATTRIBUTE_UV_0).unwrap() {
        VertexAttributeValues::Float32x2(values) => values.clone(),
        _ => panic!("UV_0 attribute is not Float32x2"),
    };

    // Shrink mesh_a's positions to be flush with mesh_b before merging
    shrink_positions_to_flush(&mut positions, &positions_b);

    let offset = positions.len() as u32;

    positions.extend_from_slice(positions_b);
    normals.extend_from_slice(normals_b);
    uvs.extend_from_slice(&uvs_b);

    // Merge indices
    let mut indices: Vec<u32> = match mesh_a.indices().unwrap() {
        Indices::U32(vec) => vec.clone(),
        _ => panic!("Only U32 indices supported"),
    };
    let indices_b: Vec<u32> = match mesh_b.indices().unwrap() {
        Indices::U32(vec) => vec.iter().map(|i| i + offset).collect(),
        _ => panic!("Only U32 indices supported"),
    };
    indices.extend(indices_b);

    // Create new mesh
    let mut mesh = Mesh::new(
        mesh_a.primitive_topology(),
        bevy::render::render_asset::RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));

    mesh
}

/// Shrinks the mesh represented by `positions` so it is flush with `other_positions` if they overlap
fn shrink_positions_to_flush(positions: &mut Vec<[f32; 3]>, other_positions: &[[f32; 3]]) {
    // Compute AABBs
    let min_a = positions.iter().fold(Vec3::splat(f32::INFINITY), |min, p| {
        Vec3::new(min.x.min(p[0]), min.y.min(p[1]), min.z.min(p[2]))
    });
    let max_a = positions
        .iter()
        .fold(Vec3::splat(f32::NEG_INFINITY), |max, p| {
            Vec3::new(max.x.max(p[0]), max.y.max(p[1]), max.z.max(p[2]))
        });
    let min_b = other_positions
        .iter()
        .fold(Vec3::splat(f32::INFINITY), |min, p| {
            Vec3::new(min.x.min(p[0]), min.y.min(p[1]), min.z.min(p[2]))
        });
    let max_b = other_positions
        .iter()
        .fold(Vec3::splat(f32::NEG_INFINITY), |max, p| {
            Vec3::new(max.x.max(p[0]), max.y.max(p[1]), max.z.max(p[2]))
        });

    // Check for overlap
    let overlap_x = max_a.x > min_b.x && min_a.x < max_b.x;
    let overlap_y = max_a.y > min_b.y && min_a.y < max_b.y;
    let overlap_z = max_a.z > min_b.z && min_a.z < max_b.z;

    if overlap_x && overlap_y && overlap_z {
        // Compute overlap amounts
        let x_overlap = (max_a.x - min_b.x).min(max_b.x - min_a.x);
        let y_overlap = (max_a.y - min_b.y).min(max_b.y - min_a.y);
        let z_overlap = (max_a.z - min_b.z).min(max_b.z - min_a.z);

        // Find axis of minimum overlap
        let (axis, amount) = {
            let mut min_axis = "x";
            let mut min_amount = x_overlap;
            if y_overlap < min_amount {
                min_axis = "y";
                min_amount = y_overlap;
            }
            if z_overlap < min_amount {
                min_axis = "z";
                min_amount = z_overlap;
            }
            (min_axis, min_amount)
        };

        // Mutate positions to shrink along that axis
        match axis {
            "x" => {
                if max_a.x - min_b.x < max_b.x - min_a.x {
                    // Shrink max_x
                    for pos in positions.iter_mut() {
                        if (pos[0] - max_a.x).abs() < 1e-4 {
                            pos[0] -= amount;
                        }
                    }
                } else {
                    // Shrink min_x
                    for pos in positions.iter_mut() {
                        if (pos[0] - min_a.x).abs() < 1e-4 {
                            pos[0] += amount;
                        }
                    }
                }
            }
            "y" => {
                if max_a.y - min_b.y < max_b.y - min_a.y {
                    for pos in positions.iter_mut() {
                        if (pos[1] - max_a.y).abs() < 1e-4 {
                            pos[1] -= amount;
                        }
                    }
                } else {
                    for pos in positions.iter_mut() {
                        if (pos[1] - min_a.y).abs() < 1e-4 {
                            pos[1] += amount;
                        }
                    }
                }
            }
            "z" => {
                if max_a.z - min_b.z < max_b.z - min_a.z {
                    for pos in positions.iter_mut() {
                        if (pos[2] - max_a.z).abs() < 1e-4 {
                            pos[2] -= amount;
                        }
                    }
                } else {
                    for pos in positions.iter_mut() {
                        if (pos[2] - min_a.z).abs() < 1e-4 {
                            pos[2] += amount;
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

#[macro_export]
macro_rules! merge_meshes {
    ($first:expr $(, $rest:expr)*) => {{
        let mut result = $first.clone(); // TODO: refactor to not use clone()?
        $(
            result = merge_meshes(&result, &$rest);
        )*
        result
    }};
}
