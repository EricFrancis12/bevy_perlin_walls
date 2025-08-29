use bevy::{
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology},
};
use bevy_inspector_egui::quick::WorldInspectorPlugin;
use noise::{NoiseFn, Perlin};

const CUBOID_WIDTH: f32 = 2.0;
const CUBOID_DEPTH: f32 = 2.0;
const CUBOID_HEIGHT: f32 = 0.2;

const RESOLUTION: usize = 20; // How many vertices per row/column on the top surface
const NOISE_SCALE: f64 = 1.0;
const NOISE_HEIGHT: f32 = 0.3;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, WorldInspectorPlugin::default()))
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Camera
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 3.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });

    // Light
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });

    // Create Perlin noise generator
    let perlin = Perlin::new(12345);

    // Generate two meshes
    let mesh1 = generate_perlin_cuboid_mesh(Vec3::new(-CUBOID_WIDTH, 0.0, 0.0), &perlin);
    let mesh2 = generate_perlin_cuboid_mesh(Vec3::new(0.0, 0.0, 0.0), &perlin);

    // Spawn mesh 1
    commands.spawn(PbrBundle {
        mesh: meshes.add(mesh1),
        material: materials.add(Color::rgb(0.6, 0.6, 1.0)),
        transform: Transform::from_translation(Vec3::ZERO),
        ..default()
    });

    // Spawn mesh 2
    commands.spawn(PbrBundle {
        mesh: meshes.add(mesh2),
        material: materials.add(Color::rgb(0.6, 1.0, 0.6)),
        transform: Transform::from_translation(Vec3::ZERO),
        ..default()
    });
}
fn generate_perlin_cuboid_mesh(origin: Vec3, perlin: &Perlin) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    let dx = CUBOID_WIDTH / RESOLUTION as f32;
    let dz = CUBOID_DEPTH / RESOLUTION as f32;

    let mut top_indices = vec![];
    let mut bottom_indices = vec![];

    // Generate top and bottom vertices
    for z in 0..=RESOLUTION {
        for x in 0..=RESOLUTION {
            let world_x = origin.x + x as f32 * dx;
            let world_z = origin.z + z as f32 * dz;

            let noise_y = perlin.get([
                world_x as f64 * NOISE_SCALE,
                0.0,
                world_z as f64 * NOISE_SCALE,
            ]) as f32;

            let top_y = origin.y + CUBOID_HEIGHT + noise_y * NOISE_HEIGHT;
            let bottom_y = origin.y;

            // Top vertex
            top_indices.push(positions.len() as u32);
            positions.push([world_x, top_y, world_z]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([x as f32 / RESOLUTION as f32, z as f32 / RESOLUTION as f32]);

            // Bottom vertex
            bottom_indices.push(positions.len() as u32);
            positions.push([world_x, bottom_y, world_z]);
            normals.push([0.0, -1.0, 0.0]);
            uvs.push([x as f32 / RESOLUTION as f32, z as f32 / RESOLUTION as f32]);
        }
    }

    // Top face indices
    for z in 0..RESOLUTION {
        for x in 0..RESOLUTION {
            let i = x + z * (RESOLUTION + 1);
            indices.extend([
                top_indices[i],
                top_indices[i + 1],
                top_indices[i + RESOLUTION + 1],
                top_indices[i + 1],
                top_indices[i + RESOLUTION + 2],
                top_indices[i + RESOLUTION + 1],
            ]);
        }
    }

    // Bottom face indices (winding reversed)
    for z in 0..RESOLUTION {
        for x in 0..RESOLUTION {
            let i = x + z * (RESOLUTION + 1);
            indices.extend([
                bottom_indices[i],
                bottom_indices[i + RESOLUTION + 1],
                bottom_indices[i + 1],
                bottom_indices[i + 1],
                bottom_indices[i + RESOLUTION + 1],
                bottom_indices[i + RESOLUTION + 2],
            ]);
        }
    }

    // Side faces (between top and bottom)
    let row_len = RESOLUTION + 1;
    for z in 0..RESOLUTION {
        for x in 0..RESOLUTION {
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
