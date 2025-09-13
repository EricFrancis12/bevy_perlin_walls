use std::f32::consts::{FRAC_PI_2, PI};

use bevy::{
    input::mouse::MouseMotion,
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
};
use bevy_inspector_egui::{bevy_egui::EguiPlugin, prelude::*, quick::WorldInspectorPlugin};
use noise::{NoiseFn, Perlin};

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

    commands.spawn((
        Name::new("Mesh 001"),
        PerlinMesh,
        Mesh3d(meshes.add(mesh_001)),
        MeshMaterial3d(materials.add(Color::srgb(0.3, 0.7, 0.9))),
        Transform::from_xyz(0.0, 0.0, CUBOID_WIDTH),
    ));

    commands.spawn((
        Name::new("Mesh 100"),
        PerlinMesh,
        Mesh3d(meshes.add(mesh_100)),
        MeshMaterial3d(materials.add(Color::srgb(0.3, 0.7, 0.9))),
        Transform::from_xyz(CUBOID_WIDTH, 0.0, 0.0),
    ));
}

fn generate_cell_mesh(origin: Vec3, perlin: &Perlin, settings: &MeshSettings) -> Mesh {
    let mesh_x_pos = generate_wall_mesh(origin, &perlin, settings, Direction::XPos);
    let mesh_x_neg = generate_wall_mesh(origin, &perlin, settings, Direction::XNeg);
    let mesh_y_pos = generate_wall_mesh(origin, &perlin, settings, Direction::YPos);
    let mesh_y_neg = generate_wall_mesh(origin, &perlin, settings, Direction::YNeg);
    let mesh_z_pos = generate_wall_mesh(origin, &perlin, settings, Direction::ZPos);
    let mesh_z_neg = generate_wall_mesh(origin, &perlin, settings, Direction::ZNeg);
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

fn generate_wall_mesh(
    origin: Vec3,
    perlin: &Perlin,
    settings: &MeshSettings,
    direction: Direction,
) -> Mesh {
    let resolution = settings.resolution;
    let noise_scale = settings.noise_scale;
    let noise_height = settings.noise_height;

    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    let dx = CUBOID_WIDTH / resolution as f32;
    let dz = CUBOID_WIDTH / resolution as f32;

    let mut top_indices = vec![];
    let mut bottom_indices = vec![];

    // Generate top and bottom vertices
    for z in 0..=resolution {
        for x in 0..=resolution {
            let world_x = x as f32 * dx;
            let world_z = z as f32 * dz;

            // Unique offset for each wall
            let wall_offset = match direction {
                Direction::YNeg => 100.0,
                Direction::XPos => 200.0,
                Direction::XNeg => 300.0,
                Direction::ZPos => 400.0,
                Direction::ZNeg => 500.0,
                Direction::YPos => 600.0,
            };

            let noise_y = perlin.get([
                (world_x as f64 + origin.x as f64) * noise_scale,
                wall_offset + origin.y as f64,
                (world_z as f64 + origin.z as f64) * noise_scale,
            ]) as f32;

            let top_y = CUBOID_DEPTH + noise_y * noise_height;
            let bottom_y = 0.;

            // Top vertex
            top_indices.push(positions.len() as u32);
            positions.push([world_x, top_y, world_z]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([x as f32 / resolution as f32, z as f32 / resolution as f32]);

            // Bottom vertex
            bottom_indices.push(positions.len() as u32);
            positions.push([world_x, bottom_y, world_z]);
            normals.push([0.0, -1.0, 0.0]);
            uvs.push([x as f32 / resolution as f32, z as f32 / resolution as f32]);
        }
    }

    // Top face
    for z in 0..resolution {
        for x in 0..resolution {
            let i = x + z * (resolution + 1);
            indices.extend([
                top_indices[i],
                top_indices[i + 1],
                top_indices[i + resolution + 1],
                top_indices[i + 1],
                top_indices[i + resolution + 2],
                top_indices[i + resolution + 1],
            ]);
        }
    }

    // Bottom face
    for z in 0..resolution {
        for x in 0..resolution {
            let i = x + z * (resolution + 1);
            indices.extend([
                bottom_indices[i],
                bottom_indices[i + resolution + 1],
                bottom_indices[i + 1],
                bottom_indices[i + 1],
                bottom_indices[i + resolution + 1],
                bottom_indices[i + resolution + 2],
            ]);
        }
    }

    // Side faces
    let row_len = resolution + 1;
    for z in 0..resolution {
        for x in 0..resolution {
            let i = x + z * row_len;
            let top0 = top_indices[i];
            let top1 = top_indices[i + 1];
            let top2 = top_indices[i + row_len];
            let top3 = top_indices[i + row_len + 1];

            let bottom0 = bottom_indices[i];
            let bottom1 = bottom_indices[i + 1];
            let bottom2 = bottom_indices[i + row_len];
            let bottom3 = bottom_indices[i + row_len + 1];

            // +X face
            indices.extend([top1, bottom1, top3, bottom1, bottom3, top3]);

            // -X face
            indices.extend([top2, bottom2, top0, bottom2, bottom0, top0]);

            // +Z face
            indices.extend([top3, bottom3, top2, bottom3, bottom2, top2]);

            // -Z face
            indices.extend([top0, bottom0, top1, bottom0, bottom1, top1]);
        }
    }

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
