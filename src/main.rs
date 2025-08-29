use bevy::{
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology},
};
use bevy_inspector_egui::{bevy_egui::EguiPlugin, prelude::*, quick::WorldInspectorPlugin};
use noise::{NoiseFn, Perlin};

const CUBOID_WIDTH: f32 = 2.0;
const CUBOID_DEPTH: f32 = 2.0;
const CUBOID_HEIGHT: f32 = 0.2;

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
            resolution: 20,
            noise_scale: 1.0,
            noise_height: 0.3,
        })
        .add_systems(Startup, setup)
        .add_systems(Update, regenerate_on_spacebar)
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

    spawn_perlin_meshes(&mut commands, &mut meshes, &mut materials, &settings);
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

    let mesh1 = generate_perlin_cuboid_mesh(Vec3::new(-CUBOID_WIDTH, 0.0, 0.0), &perlin, settings);
    let mesh2 = generate_perlin_cuboid_mesh(Vec3::new(0.0, 0.0, 0.0), &perlin, settings);

    commands.spawn((
        Name::new("Mesh 1"),
        PerlinMesh,
        Mesh3d(meshes.add(mesh1)),
        MeshMaterial3d(materials.add(Color::srgb(0.6, 0.6, 1.0))),
        Transform::default(),
    ));

    commands.spawn((
        Name::new("Mesh 2"),
        PerlinMesh,
        Mesh3d(meshes.add(mesh2)),
        MeshMaterial3d(materials.add(Color::srgb(0.6, 1.0, 0.6))),
        Transform::default(),
    ));
}

fn generate_perlin_cuboid_mesh(origin: Vec3, perlin: &Perlin, settings: &MeshSettings) -> Mesh {
    let resolution = settings.resolution;
    let noise_scale = settings.noise_scale;
    let noise_height = settings.noise_height;

    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    let dx = CUBOID_WIDTH / resolution as f32;
    let dz = CUBOID_DEPTH / resolution as f32;

    let mut top_indices = vec![];
    let mut bottom_indices = vec![];

    // Generate top and bottom vertices
    for z in 0..=resolution {
        for x in 0..=resolution {
            let world_x = origin.x + x as f32 * dx;
            let world_z = origin.z + z as f32 * dz;

            let noise_y = perlin.get([
                world_x as f64 * noise_scale,
                0.0,
                world_z as f64 * noise_scale,
            ]) as f32;

            let top_y = origin.y + CUBOID_HEIGHT + noise_y * noise_height;
            let bottom_y = origin.y;

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
