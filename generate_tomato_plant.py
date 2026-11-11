"""
===============================================================================
GENERADOR BOTÁNICO DE PLANTA DE TOMATE CON TOMATES SEPARADOS Y ELEVADOS
===============================================================================
Este script genera una tomatera procedural fotorrealista y anatómicamente correcta
con físicas PhysX en NVIDIA Isaac Sim para pruebas de manipulación robótica.

CARACTERÍSTICAS DE DISTRIBUCIÓN:
1. Tomates Separados e Independientes (Sin Racimos Apretados):
   - Cada tomate nace de su propio pedúnculo/pedicelo en nudos independientes del tallo.
   - Distribución radial en espiral dorada (137.5°) para evitar colisiones entre frutos.
2. Mayor Altura (Especial para Manipulación Robótica):
   - Los frutos se ubican en la mitad/tercio superior del tallo (nudos 6 a 13),
     a alturas entre Z = 0.50m y Z = 0.85m sobre el suelo.
3. Conexión Orgánica y Físicas PhysX:
   - Tallo cónico y curvado en S.
   - Pecíolos y folíolos articulados.
   - Cáliz envuelto en la parte superior del fruto.
   - Uniones de desprendimiento en la zona de absición (PhysX FixedJoint = 5.0 N).
   - Gradiente de maduración (maduros abajo, verdes arriba).

INSTRUCCIONES DE EJECUCIÓN:
- Ejecutar directamente desde Isaac Sim (Script Editor o VS Code Extension).
===============================================================================
"""

import math
import os
import random
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema

# ==============================================================================
# PARAMETERS - CONFIGURACIÓN GENERAL Y BOTÁNICA (MODIFICABLE POR EL USUARIO)
# ==============================================================================

# --- Escena y Suelo ---
ground_size = 10.0          # Tamaño del plano de suelo en metros (10.0m x 10.0m = 100 m²)
gravity_magnitude = 9.81    # Magnitud de la gravedad (m/s²)

# --- Maceta ---
pot_radius = 0.15           # Radio de la maceta (m)
pot_height = 0.20           # Altura de la maceta (m)
pot_mass = 2.0              # Masa de la maceta (kg)


def create_procedural_leaf_mesh(stage, leaf_path, scale_f, color_leaf):
    """
    Crea una malla 3D (UsdGeom.Mesh) botánicamente realista con silueta lanceolada,
    bordes lobulados y curvatura orgánica 3D para cada folíolo de tomate.
    """
    leaf_mesh = UsdGeom.Mesh.Define(stage, leaf_path)
    
    length = 0.075 * scale_f
    width = 0.038 * scale_f
    curl = 0.008 * scale_f
    
    # Vértices 3D que forman la curvatura parabólica y los lóbulos lanceolados de una hoja de tomate
    points = [
        # Nervio central (Spine / Midrib)
        Gf.Vec3f(0.0, 0.0, 0.0),                   # 0: Base
        Gf.Vec3f(0.0, length * 0.35, -curl * 0.5),  # 1: Centro-inferior
        Gf.Vec3f(0.0, length * 0.70, -curl * 0.8),  # 2: Centro-superior
        Gf.Vec3f(0.0, length * 1.0,  -curl * 0.3),  # 3: Ápice (Punta)
        
        # Borde Izquierdo
        Gf.Vec3f(-width * 0.4, length * 0.20,  curl * 0.3),  # 4: Lóbulo base izq
        Gf.Vec3f(-width * 0.5, length * 0.45,  curl * 0.6),  # 5: Ancho máximo izq
        Gf.Vec3f(-width * 0.35, length * 0.75, curl * 0.4),  # 6: Lóbulo sup izq
        
        # Borde Derecho
        Gf.Vec3f(width * 0.4, length * 0.20,   curl * 0.3),  # 7: Lóbulo base der
        Gf.Vec3f(width * 0.5, length * 0.45,   curl * 0.6),  # 8: Ancho máximo der
        Gf.Vec3f(width * 0.35, length * 0.75,  curl * 0.4),  # 9: Lóbulo sup der
    ]
    
    # 8 Caras poligonales construyendo la lámina foliar orgánica de doble cara
    faceVertexCounts = [3, 4, 3, 3, 3, 4, 3, 3]
    faceVertexIndices = [
        # Lado Izquierdo
        0, 4, 1,
        1, 4, 5, 2,
        2, 5, 6,
        2, 6, 3,
        # Lado Derecho
        0, 1, 7,
        1, 8, 7,
        1, 2, 8,
        2, 9, 8,
        2, 3, 9
    ]
    
    leaf_mesh.CreatePointsAttr().Set(points)
    leaf_mesh.CreateFaceVertexCountsAttr().Set(faceVertexCounts)
    leaf_mesh.CreateFaceVertexIndicesAttr().Set(faceVertexIndices)
    leaf_mesh.CreateDoubleSidedAttr().Set(True)
    leaf_mesh.CreateDisplayColorAttr().Set([color_leaf])
    
    leaf_mesh.CreateExtentAttr().Set([
        Gf.Vec3f(-width, 0.0, -curl),
        Gf.Vec3f(width, length, curl)
    ])
    
    return leaf_mesh


# ==============================================================================
# PARAMETERS - CONFIGURACIÓN GENERAL Y BOTÁNICA (MODIFICABLE POR EL USUARIO)
# ==============================================================================
# --- Planta y Tutor ---
plant_height = 0.70         # Altura total de la planta (m)
plant_mass = 0.60           # Masa total de la estructura vegetal (kg)
stem_radius_base = 0.011    # Radio del tallo en la base (11 mm)
stem_radius_top = 0.0035    # Radio del tallo en el ápice (3.5 mm)
num_stem_segments = 14      # Número de segmentos para curvatura orgánica del tallo
enable_secondary_branches = True  # Habilitado: genera ramas secundarias con hojas orgánicas 3D (Mesh)

# --- Tomates Individuales Elevados ---
num_tomatoes = 8            # Número de tomates independientes

# --- Físicas de Desprendimiento (PhysX Breakable Joint) ---
stem_break_force = 18.0      # Fuerza de desprendimiento en Newtons (PhysX breakForce)
stem_break_torque = 1.2      # Torque de desprendimiento en N·m (PhysX breakTorque)

# --- Robots Franka Research 3 (FR3) ---
enable_robot = True         # Habilitar la inclusión de robots Franka en la escena
num_robots = 1              # Número de robots por entorno (1 o 2)
robot_distance_x = 0.70     # Distancia simétrica X al centro de la planta (70 cm)

# --- Cesta Agrícola de Recolección (Equidistante) ---
enable_basket = True        # Habilitar la inclusión de la cesta agrícola de cosecha
basket_pos_x = 0.0          # Posición X centrada equidistante entre ambos robots (0.0 m)
basket_pos_y = 0.55         # Posición Y frente a la maceta (45 cm)
basket_pos_z = 0.0          # Posición Z apoyada en el suelo

# --- Generación Procedural y Exportación ---
random_seed = 42            # Semilla aleatoria para reproducibilidad exacta
output_usd_path = "tomato_plant.usd" # Ruta del archivo USD a guardar

# --- Paleta de Colores Orgánica PBR (RGB 0.0 - 1.0) ---
color_pot_rgb = (0.58, 0.28, 0.14)        # Terracota
color_stem_rgb = (0.16, 0.42, 0.12)       # Verde tallo vegetal
color_leaf_rgb = (0.18, 0.52, 0.15)       # Verde hoja silvestre
color_calyx_rgb = (0.20, 0.56, 0.16)      # Verde sépalo y pedicelo
color_wood_rgb = (0.45, 0.32, 0.18)       # Tutor de madera agrícola
color_ring_rgb = (0.12, 0.16, 0.12)       # Anillas de sujeción (plástico oscuro)


# ==============================================================================
# AUXILIARES MATEMÁTICOS Y DE TRANSFORMACIÓN USD
# ==============================================================================

def get_rotation_between(v1, v2):
    """Calcula el cuaternión Gf.Quatf para orientar el vector v1 hacia el vector v2."""
    v1 = v1.GetNormalized()
    v2 = v2.GetNormalized()
    dot = Gf.Dot(v1, v2)
    if dot >= 0.99999:
        return Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    elif dot <= -0.99999:
        return Gf.Quatf(0.0, 1.0, 0.0, 0.0)
    
    axis = Gf.Cross(v1, v2).GetNormalized()
    angle_rad = math.acos(max(-1.0, min(1.0, dot)))
    s = math.sin(angle_rad / 2.0)
    c = math.cos(angle_rad / 2.0)
    return Gf.Quatf(c, axis[0] * s, axis[1] * s, axis[2] * s)


def setup_segment_xform(prim_or_gprim, p0, p1):
    """Orienta y posiciona un cilindro primario para que se extienda exactamente entre p0 y p1."""
    mid = (p0 + p1) * 0.5
    dir_vec = p1 - p0
    length = dir_vec.GetLength()
    if length < 1e-6:
        dir_vec = Gf.Vec3d(0, 0, 1)
        length = 1e-6
    else:
        dir_vec = dir_vec.GetNormalized()
    
    quat = get_rotation_between(Gf.Vec3d(0, 0, 1), dir_vec)
    
    xformable = UsdGeom.Xformable(prim_or_gprim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(mid)
    xformable.AddOrientOp().Set(quat)
    return length


def setup_xform(prim_or_gprim, translate=None, rotate_xyz=None, scale=None):
    """Configura transformaciones estándar de forma limpia."""
    xformable = UsdGeom.Xformable(prim_or_gprim)
    xformable.ClearXformOpOrder()
    if translate is not None:
        xformable.AddTranslateOp().Set(translate)
    if rotate_xyz is not None:
        xformable.AddRotateXYZOp().Set(rotate_xyz)
    if scale is not None:
        xformable.AddScaleOp().Set(scale)


def create_procedural_harvest_basket(stage, basket_path, position):
    """
    Crea una caja agrícola de recolección HUECA con paredes trapezoidales inclinadas,
    borde superior reforzado (rim) y asas laterales de transporte.
    """
    if stage.GetPrimAtPath(basket_path).IsValid():
        stage.RemovePrim(basket_path)

    print(f"[Info] Generando cesta agrícola de cosecha en {position}...")
    basket_xform = UsdGeom.Xform.Define(stage, basket_path)
    setup_xform(basket_xform, translate=Gf.Vec3d(*position))

    color_basket = Gf.Vec3f(0.12, 0.42, 0.82)  # Azul agrícola vibrante
    color_handle = Gf.Vec3f(0.08, 0.28, 0.58)  # Azul oscuro para borde y asas

    b_base_w = 0.28   # Ancho base (28 cm)
    b_base_d = 0.28   # Profundidad base (28 cm)
    b_top_w = 0.36    # Ancho boca superior (36 cm, forma trapezoidal)
    b_top_d = 0.36    # Profundidad boca superior (36 cm)
    h_size = 0.22     # Altura de paredes (22 cm)
    wall_t = 0.014    # Grosor de pared (1.4 cm)

    # 1. Base del fondo (Floor plate)
    floor_path = f"{basket_path}/Floor"
    floor_cube = UsdGeom.Cube.Define(stage, floor_path)
    floor_cube.CreateSizeAttr().Set(1.0)
    floor_cube.CreateDisplayColorAttr().Set([color_basket])
    setup_xform(floor_cube, translate=Gf.Vec3d(0.0, 0.0, wall_t / 2.0), scale=Gf.Vec3f(b_base_w, b_base_d, wall_t))
    UsdPhysics.CollisionAPI.Apply(floor_cube.GetPrim())

    # 2. Paredes Inclinadas Trapezoidales (Norte +Y, Sur -Y, Este +X, Oeste -X)
    pitch_angle = 8.0 # Grados de apertura cónica hacia arriba

    # Pared Norte (+Y)
    w_north = UsdGeom.Cube.Define(stage, f"{basket_path}/Wall_North")
    w_north.CreateSizeAttr().Set(1.0)
    w_north.CreateDisplayColorAttr().Set([color_basket])
    pos_n = Gf.Vec3d(0.0, (b_base_d + b_top_d) / 4.0 - wall_t / 2.0, h_size / 2.0)
    rot_n = Gf.Vec3f(-pitch_angle, 0.0, 0.0)
    scale_n = Gf.Vec3f(b_top_w, wall_t, h_size)
    setup_xform(w_north, translate=pos_n, rotate_xyz=rot_n, scale=scale_n)
    UsdPhysics.CollisionAPI.Apply(w_north.GetPrim())

    # Pared Sur (-Y)
    w_south = UsdGeom.Cube.Define(stage, f"{basket_path}/Wall_South")
    w_south.CreateSizeAttr().Set(1.0)
    w_south.CreateDisplayColorAttr().Set([color_basket])
    pos_s = Gf.Vec3d(0.0, -(b_base_d + b_top_d) / 4.0 + wall_t / 2.0, h_size / 2.0)
    rot_s = Gf.Vec3f(pitch_angle, 0.0, 0.0)
    scale_s = Gf.Vec3f(b_top_w, wall_t, h_size)
    setup_xform(w_south, translate=pos_s, rotate_xyz=rot_s, scale=scale_s)
    UsdPhysics.CollisionAPI.Apply(w_south.GetPrim())

    # Pared Este (+X)
    w_east = UsdGeom.Cube.Define(stage, f"{basket_path}/Wall_East")
    w_east.CreateSizeAttr().Set(1.0)
    w_east.CreateDisplayColorAttr().Set([color_basket])
    pos_e = Gf.Vec3d((b_base_w + b_top_w) / 4.0 - wall_t / 2.0, 0.0, h_size / 2.0)
    rot_e = Gf.Vec3f(0.0, pitch_angle, 0.0)
    scale_e = Gf.Vec3f(wall_t, b_top_d - 2 * wall_t, h_size)
    setup_xform(w_east, translate=pos_e, rotate_xyz=rot_e, scale=scale_e)
    UsdPhysics.CollisionAPI.Apply(w_east.GetPrim())

    # Pared Oeste (-X)
    w_west = UsdGeom.Cube.Define(stage, f"{basket_path}/Wall_West")
    w_west.CreateSizeAttr().Set(1.0)
    w_west.CreateDisplayColorAttr().Set([color_basket])
    pos_w = Gf.Vec3d(-(b_base_w + b_top_w) / 4.0 + wall_t / 2.0, 0.0, h_size / 2.0)
    rot_w = Gf.Vec3f(0.0, -pitch_angle, 0.0)
    scale_w = Gf.Vec3f(wall_t, b_top_d - 2 * wall_t, h_size)
    setup_xform(w_west, translate=pos_w, rotate_xyz=rot_w, scale=scale_w)
    UsdPhysics.CollisionAPI.Apply(w_west.GetPrim())

    # 3. Borde Reforzado Superior Perimetral HUECO (4 tiras de labio sin tapa superior)
    rim_t = 0.02 # Grosor del labio perimetral
    rim_z = h_size - 0.008
    
    rim_n = UsdGeom.Cube.Define(stage, f"{basket_path}/TopRim_North")
    rim_n.CreateSizeAttr().Set(1.0)
    rim_n.CreateDisplayColorAttr().Set([color_handle])
    setup_xform(rim_n, translate=Gf.Vec3d(0.0, b_top_d / 2.0, rim_z), scale=Gf.Vec3f(b_top_w + 0.03, rim_t, 0.016))
    UsdPhysics.CollisionAPI.Apply(rim_n.GetPrim())

    rim_s = UsdGeom.Cube.Define(stage, f"{basket_path}/TopRim_South")
    rim_s.CreateSizeAttr().Set(1.0)
    rim_s.CreateDisplayColorAttr().Set([color_handle])
    setup_xform(rim_s, translate=Gf.Vec3d(0.0, -b_top_d / 2.0, rim_z), scale=Gf.Vec3f(b_top_w + 0.03, rim_t, 0.016))
    UsdPhysics.CollisionAPI.Apply(rim_s.GetPrim())

    rim_e = UsdGeom.Cube.Define(stage, f"{basket_path}/TopRim_East")
    rim_e.CreateSizeAttr().Set(1.0)
    rim_e.CreateDisplayColorAttr().Set([color_handle])
    setup_xform(rim_e, translate=Gf.Vec3d(b_top_w / 2.0, 0.0, rim_z), scale=Gf.Vec3f(rim_t, b_top_d + 0.03, 0.016))
    UsdPhysics.CollisionAPI.Apply(rim_e.GetPrim())

    rim_w = UsdGeom.Cube.Define(stage, f"{basket_path}/TopRim_West")
    rim_w.CreateSizeAttr().Set(1.0)
    rim_w.CreateDisplayColorAttr().Set([color_handle])
    setup_xform(rim_w, translate=Gf.Vec3d(-b_top_w / 2.0, 0.0, rim_z), scale=Gf.Vec3f(rim_t, b_top_d + 0.03, 0.016))
    UsdPhysics.CollisionAPI.Apply(rim_w.GetPrim())

    # 4. Asas Laterales de Transporte (Handles)
    for s_side, offset_y in [("Left", (b_top_d / 2.0) + 0.025), ("Right", -(b_top_d / 2.0) - 0.025)]:
        handle = UsdGeom.Cylinder.Define(stage, f"{basket_path}/Handle_{s_side}")
        handle.CreateRadiusAttr().Set(0.009)
        handle.CreateHeightAttr().Set(0.12)
        handle.CreateAxisAttr().Set("X")
        handle.CreateDisplayColorAttr().Set([color_handle])
        setup_xform(handle, translate=Gf.Vec3d(0.0, offset_y, h_size - 0.01))


# ==============================================================================
# GENERACIÓN BOTÁNICA PRINCIPAL
# ==============================================================================

def create_procedural_tomato_plant(env_origin=Gf.Vec3d(0, 0, 0), env_idx=0, num_robots=1):
    random.seed(random_seed)

    import isaacsim.core.experimental.utils.app as app_utils

    # Detener la simulación limpiamente antes de modificar o borrar la escena USD
    app_utils.stop()
    app_utils.update_app(steps=2)

    # --------------------------------------------------------------------------
    # 1. INICIALIZACIÓN DE ESCENA Y PLANO DE SUELO
    # --------------------------------------------------------------------------
    stage = None
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
    except Exception:
        pass

    if stage is None or not stage:
        print("[Info] Creando un nuevo stage USD en memoria...")
        stage = Usd.Stage.CreateInMemory()

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world_path = f"/World/Env_{env_idx}"
    if not stage.GetPrimAtPath(world_path).IsValid():
        UsdGeom.Xform.Define(stage, world_path)

    # Limpiar ejecuciones previas en este scope
    for name in ["Pot", "Plant", "Franka", "Franka_Front", "Franka_Back", "HarvestBasket"]:
        if stage.GetPrimAtPath(f"{world_path}/{name}").IsValid():
            stage.RemovePrim(f"{world_path}/{name}")

    # Escena Física PhysX
    physics_scene_path = "/World/PhysicsScene"
    if not stage.GetPrimAtPath(physics_scene_path).IsValid():
        scene = UsdPhysics.Scene.Define(stage, physics_scene_path)
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(gravity_magnitude)

    # Plano de suelo estático (GroundPlane con textura visual de rejilla y físicas PhysX)
    ground_path = "/World/GroundPlane"
    if not stage.GetPrimAtPath(ground_path).IsValid():
        try:
            import omni.kit.commands
            omni.kit.commands.execute(
                "AddGroundPlaneCommand",
                stage=stage,
                planePath=ground_path,
                axis="Z",
                size=ground_size,
                position=Gf.Vec3f(0.0, 0.0, 0.0),
                color=Gf.Vec3f(0.5, 0.5, 0.5)
            )
            print("[Éxito] Plano de suelo (GroundPlane) con rejilla de Isaac Sim añadido correctamente.")
        except Exception:
            ground = UsdGeom.Plane.Define(stage, ground_path)
            ground.CreateAxisAttr().Set("Z")
            ground.CreateSizeAttr().Set(ground_size)
            UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    # --------------------------------------------------------------------------
    # 2. CONTENEDORES Y MACETA DINÁMICA (Anclada a env_origin)
    # --------------------------------------------------------------------------
    pot_path = f"{world_path}/Pot" # Semantic Containers: gathers pot+plant in a single element. Does not have geometry.
    pot_mesh_path = f"{pot_path}/PotMesh" # Pot 3D geometry.
    plant_path = f"{pot_path}/Plant"
    main_stem_path = f"{plant_path}/MainStem"
    leaves_path = f"{plant_path}/Leaves"
    pedicels_path = f"{plant_path}/Pedicels"
    tomatoes_path = f"{plant_path}/Tomatoes"
    joints_path = f"{plant_path}/Joints"

    pot_xform = UsdGeom.Xform.Define(stage, pot_path)
    setup_xform(pot_xform, translate=env_origin)

    plant_xform = UsdGeom.Xform.Define(stage, plant_path)
    UsdGeom.Xform.Define(stage, main_stem_path)
    UsdGeom.Xform.Define(stage, leaves_path)
    UsdGeom.Xform.Define(stage, pedicels_path)
    UsdGeom.Xform.Define(stage, tomatoes_path)
    UsdGeom.Xform.Define(stage, joints_path)

    # Offset relativo local respecto al centro de Pot (se coloca en la boca de la maceta)
    setup_xform(plant_xform, translate=Gf.Vec3d(0.0, 0.0, pot_height))

    # Planta como cuerpo rígido dinámico
    plant_prim = plant_xform.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(plant_prim)
    plant_mass_api = UsdPhysics.MassAPI.Apply(plant_prim)
    plant_mass_api.CreateMassAttr().Set(plant_mass)

    color_pot = Gf.Vec3f(*color_pot_rgb)
    color_stem = Gf.Vec3f(*color_stem_rgb)
    color_leaf = Gf.Vec3f(*color_leaf_rgb)
    color_calyx = Gf.Vec3f(*color_calyx_rgb)
    color_wood = Gf.Vec3f(*color_wood_rgb)
    color_ring = Gf.Vec3f(*color_ring_rgb)

    # --------------------------------------------------------------------------
    # MACETA AGRÍCOLA DE TERRACOTA (Cuerpo principal, labio superior y tierra fértil)
    # --------------------------------------------------------------------------
    pot_cyl = UsdGeom.Cylinder.Define(stage, pot_mesh_path)
    pot_cyl.CreateHeightAttr().Set(pot_height)
    pot_cyl.CreateRadiusAttr().Set(pot_radius)
    pot_cyl.CreateAxisAttr().Set("Z")
    pot_cyl.CreateDisplayColorAttr().Set([color_pot])
    setup_xform(pot_cyl, translate=Gf.Vec3d(0.0, 0.0, pot_height / 2.0))
    
    pot_prim = pot_cyl.GetPrim()
    UsdPhysics.CollisionAPI.Apply(pot_prim)

    # Labio superior prominente de la maceta (Rim)
    pot_rim = UsdGeom.Cylinder.Define(stage, f"{pot_path}/PotRim")
    pot_rim.CreateHeightAttr().Set(pot_height * 0.15)
    pot_rim.CreateRadiusAttr().Set(pot_radius * 1.08)
    pot_rim.CreateAxisAttr().Set("Z")
    pot_rim.CreateDisplayColorAttr().Set([color_pot])
    setup_xform(pot_rim, translate=Gf.Vec3d(0.0, 0.0, pot_height * 0.925))

    # Capa de tierra agrícola oscura fértil
    soil_cyl = UsdGeom.Cylinder.Define(stage, f"{pot_path}/Soil")
    soil_cyl.CreateHeightAttr().Set(0.015)
    soil_cyl.CreateRadiusAttr().Set(pot_radius * 0.97)
    soil_cyl.CreateAxisAttr().Set("Z")
    soil_cyl.CreateDisplayColorAttr().Set([Gf.Vec3f(0.15, 0.10, 0.06)])
    setup_xform(soil_cyl, translate=Gf.Vec3d(0.0, 0.0, pot_height * 0.98))

    # Anclaje maceta - planta
    pot_plant_joint = UsdPhysics.FixedJoint.Define(stage, f"{joints_path}/PotPlantJoint")
    pot_plant_joint.CreateBody0Rel().SetTargets([Sdf.Path(pot_mesh_path)])
    pot_plant_joint.CreateBody1Rel().SetTargets([Sdf.Path(plant_path)])
    pot_plant_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, pot_height / 2.0))
    pot_plant_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    pot_plant_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    pot_plant_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    PhysxSchema.PhysxJointAPI.Apply(pot_plant_joint.GetPrim())

    # --------------------------------------------------------------------------
    # 3. TALLO PRINCIPAL ORGÁNICO SEGMENTADO (Curvado y Cónico)
    # --------------------------------------------------------------------------
    print("[Info] Generando tallo principal orgánico...")
    stem_nodes = []
    stem_tangents = []
    
    current_pos = Gf.Vec3d(0.0, 0.0, 0.0)
    stem_nodes.append(current_pos)
    
    for i in range(num_stem_segments):
        t = (i + 1) / float(num_stem_segments)
        
        bend_x = 0.018 * math.sin(t * math.pi * 1.5) + random.uniform(-0.003, 0.003)
        bend_y = 0.012 * math.cos(t * math.pi * 2.0) + random.uniform(-0.003, 0.003)
        next_pos = Gf.Vec3d(bend_x, bend_y, t * plant_height)
        
        r_seg = stem_radius_base * (1.0 - 0.65 * t) + stem_radius_top * (0.65 * t)
        
        seg_path = f"{main_stem_path}/Segment_{i:02d}"
        seg_cyl = UsdGeom.Cylinder.Define(stage, seg_path)
        seg_cyl.CreateRadiusAttr().Set(r_seg)
        seg_cyl.CreateAxisAttr().Set("Z")
        seg_cyl.CreateDisplayColorAttr().Set([color_stem])
        
        h_actual = setup_segment_xform(seg_cyl, current_pos, next_pos)
        seg_cyl.CreateHeightAttr().Set(h_actual)
        UsdPhysics.CollisionAPI.Apply(seg_cyl.GetPrim())

        tangent = (next_pos - current_pos).GetNormalized()
        stem_nodes.append(next_pos)
        stem_tangents.append(tangent)
        current_pos = next_pos

    # --------------------------------------------------------------------------
    # 4. TUTOR DE MADERA AGRÍCOLA Y ANILLAS DE SUJECIÓN
    # --------------------------------------------------------------------------
    stake_path = f"{plant_path}/SupportStake"
    stake_cyl = UsdGeom.Cylinder.Define(stage, stake_path)
    stake_cyl.CreateHeightAttr().Set(plant_height * 1.05)
    stake_cyl.CreateRadiusAttr().Set(0.009)
    stake_cyl.CreateAxisAttr().Set("Z")
    stake_cyl.CreateDisplayColorAttr().Set([color_wood])
    
    stake_offset = Gf.Vec3d(-0.025, 0.015, (plant_height * 1.05) / 2.0)
    setup_xform(stake_cyl, translate=stake_offset)
    UsdPhysics.CollisionAPI.Apply(stake_cyl.GetPrim())

    for ring_idx, node_i in enumerate([3, 7, 11]):
        ring_p = stem_nodes[node_i]
        ring_path = f"{plant_path}/Ring_{ring_idx:02d}"
        ring_cyl = UsdGeom.Cylinder.Define(stage, ring_path)
        ring_cyl.CreateHeightAttr().Set(0.012)
        ring_cyl.CreateRadiusAttr().Set(0.016)
        ring_cyl.CreateAxisAttr().Set("Z")
        ring_cyl.CreateDisplayColorAttr().Set([color_ring])
        setup_xform(ring_cyl, translate=Gf.Vec3d(ring_p[0] - 0.012, ring_p[1] + 0.007, ring_p[2]))

    # --------------------------------------------------------------------------
    # 5. HOJAS COMPUESTAS BOTÁNICAS CON FOLÍOLOS (RAMAS SECUNDARIAS)
    # --------------------------------------------------------------------------
    if enable_secondary_branches:
        print("[Info] Generando hojas compuestas distribuidas...")
        leaf_node_indices = [1, 2, 4, 5, 7, 8, 10, 11, 12]

        for l_idx, node_i in enumerate(leaf_node_indices):
            n_pos = stem_nodes[node_i]
            
            yaw_deg = (l_idx * 137.5 + random.uniform(-10, 10)) % 360.0
            pitch_deg = random.uniform(35.0, 50.0)
            
            yaw_rad = math.radians(yaw_deg)
            pitch_rad = math.radians(pitch_deg)
            
            petiole_len = random.uniform(0.18, 0.24)
            dir_x = math.sin(pitch_rad) * math.cos(yaw_rad)
            dir_y = math.sin(pitch_rad) * math.sin(yaw_rad)
            dir_z = math.cos(pitch_rad) - 0.15
            
            petiole_end = n_pos + Gf.Vec3d(dir_x, dir_y, dir_z) * petiole_len
            
            # Deformidad y curvatura orgánica 3D (2 segmentos con ondulación sin decaer hacia el suelo)
            petiole_mid = (n_pos + petiole_end) * 0.5 + Gf.Vec3d(
                random.uniform(-0.015, 0.015),
                random.uniform(-0.015, 0.015),
                random.uniform(0.008, 0.022)
            )
            
            leaf_group_path = f"{leaves_path}/Leaf_{l_idx:02d}"
            leaf_xform = UsdGeom.Xform.Define(stage, leaf_group_path)
            setup_xform(leaf_xform, translate=n_pos)

            # Segmento 1 del pecíolo (relativo al nodo)
            p_cyl1_path = f"{leaf_group_path}/Petiole_Seg1"
            p_cyl1 = UsdGeom.Cylinder.Define(stage, p_cyl1_path)
            p_cyl1.CreateRadiusAttr().Set(0.0038)
            p_cyl1.CreateDisplayColorAttr().Set([color_stem])
            h_pet1 = setup_segment_xform(p_cyl1, Gf.Vec3d(0, 0, 0), petiole_mid - n_pos)
            p_cyl1.CreateHeightAttr().Set(h_pet1)

            # Segmento 2 del pecíolo (relativo al nodo)
            p_cyl2_path = f"{leaf_group_path}/Petiole_Seg2"
            p_cyl2 = UsdGeom.Cylinder.Define(stage, p_cyl2_path)
            p_cyl2.CreateRadiusAttr().Set(0.0030)
            p_cyl2.CreateDisplayColorAttr().Set([color_stem])
            h_pet2 = setup_segment_xform(p_cyl2, petiole_mid - n_pos, petiole_end - n_pos)
            p_cyl2.CreateHeightAttr().Set(h_pet2)

            leaflet_positions = [
                (petiole_end - n_pos, 1.0, 0.0),
                ((petiole_mid + (petiole_end - petiole_mid) * 0.50) - n_pos, 0.85, 45.0),
                ((petiole_mid + (petiole_end - petiole_mid) * 0.50) - n_pos, 0.85, -45.0),
                ((petiole_mid - n_pos) * 0.50, 0.70, 50.0),
                ((petiole_mid - n_pos) * 0.50, 0.70, -50.0),
            ]

            for lf_i, (lf_pos, scale_f, side_angle) in enumerate(leaflet_positions):
                lf_path = f"{leaf_group_path}/Leaflet_{lf_i:02d}"
                lf_mesh = create_procedural_leaf_mesh(stage, lf_path, scale_f, color_leaf)
                lf_rot = Gf.Vec3f(random.uniform(5, 20), side_angle, yaw_deg)
                setup_xform(lf_mesh, translate=lf_pos, rotate_xyz=lf_rot)
                
                col_api = UsdPhysics.CollisionAPI.Apply(lf_mesh.GetPrim())
                col_api.CreateCollisionEnabledAttr().Set(False)
                mesh_col = UsdPhysics.MeshCollisionAPI.Apply(lf_mesh.GetPrim())
                mesh_col.CreateApproximationAttr().Set("convexHull")
    else:
        print("[Info] Ramas secundarias (hojas sin tomates) desactivadas.")

    # --------------------------------------------------------------------------
    # 6. TOMATES INDIVIDUALES ELEVADOS Y SEPARADOS EN SECTORES ALCANZABLES (FRONTAL Y LATERALES)
    # --------------------------------------------------------------------------
    print("[Info] Generando 10 tomates independientes en sectores Frontal, Lateral Norte (+Y) y Lateral Sur (-Y)...")
    
    tomato_configs = [
        # (node_index, yaw_degrees, out_distance)
        (5,  180.0, 0.11),  # Tomato 01: Frontal (-X)
        (6,  125.0, 0.12),  # Tomato 02: Lateral Norte (+Y)
        (7,  235.0, 0.12),  # Tomato 03: Lateral Sur (-Y)
        (8,  165.0, 0.10),  # Tomato 04: Frontal (-X)
        (9,  115.0, 0.13),  # Tomato 05: Lateral Norte (+Y)
        (10, 245.0, 0.13),  # Tomato 06: Lateral Sur (-Y)
        (11, 195.0, 0.11),  # Tomato 07: Frontal (-X)
        (12, 135.0, 0.10),  # Tomato 08: Lateral Norte (+Y)
        (13, 225.0, 0.10),  # Tomato 09: Lateral Sur (-Y)
        (14, 180.0, 0.12),  # Tomato 10: Frontal (-X)
    ]

    for idx, (node_i, yaw_deg, out_dist) in enumerate(tomato_configs):
        tom_name = f"Tomato_{idx+1:02d}"
        t_path = f"{tomatoes_path}/{tom_name}"

        node_p = stem_nodes[node_i]
        
        progress = idx / float(len(tomato_configs) - 1)
        if progress < 0.4:
            tom_color_rgb = (0.86, 0.05, 0.03)
            tom_radius = 0.022
            tom_mass = 0.04
        elif progress < 0.75:
            tom_color_rgb = (0.92, 0.28, 0.04)
            tom_radius = 0.019
            tom_mass = 0.03
        else:
            tom_color_rgb = (0.55, 0.70, 0.15)
            tom_radius = 0.015
            tom_mass = 0.02

        tom_color = Gf.Vec3f(*tom_color_rgb)

        yaw_rad = math.radians(yaw_deg)
        dir_out = Gf.Vec3d(math.cos(yaw_rad), math.sin(yaw_rad), -0.15).GetNormalized()
        
        pedicel_start = node_p
        pedicel_end = pedicel_start + dir_out * out_dist + Gf.Vec3d(0.0, 0.0, -0.015)
        
        # Deformidad orgánica 3D en la rama del tomate (Punto medio sinusal)
        pedicel_mid = (pedicel_start + pedicel_end) * 0.5 + Gf.Vec3d(
            random.uniform(-0.012, 0.012),
            random.uniform(-0.012, 0.012),
            random.uniform(0.006, 0.016)
        )

        # A. CILINDROS SEGMENTADOS DEL PEDICELO (Deformidad Curva)
        pedicel_path1 = f"{pedicels_path}/Pedicel_{idx+1:02d}_Seg1"
        ped_cyl1 = UsdGeom.Cylinder.Define(stage, pedicel_path1)
        ped_cyl1.CreateRadiusAttr().Set(0.0028)
        ped_cyl1.CreateDisplayColorAttr().Set([color_calyx])
        h_ped1 = setup_segment_xform(ped_cyl1, pedicel_start, pedicel_mid)
        ped_cyl1.CreateHeightAttr().Set(h_ped1)
        UsdPhysics.CollisionAPI.Apply(ped_cyl1.GetPrim())

        pedicel_path2 = f"{pedicels_path}/Pedicel_{idx+1:02d}_Seg2"
        ped_cyl2 = UsdGeom.Cylinder.Define(stage, pedicel_path2)
        ped_cyl2.CreateRadiusAttr().Set(0.0022)
        ped_cyl2.CreateDisplayColorAttr().Set([color_calyx])
        h_ped2 = setup_segment_xform(ped_cyl2, pedicel_mid, pedicel_end)
        ped_cyl2.CreateHeightAttr().Set(h_ped2)
        UsdPhysics.CollisionAPI.Apply(ped_cyl2.GetPrim())

        # B. CONTENEDOR PRINCIPAL DEL TOMATE (RIGID BODY)
        tom_xform = UsdGeom.Xform.Define(stage, t_path)
        setup_xform(tom_xform, translate=pedicel_end)
        
        tom_prim = tom_xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(tom_prim)
        mass_api = UsdPhysics.MassAPI.Apply(tom_prim)
        mass_api.CreateMassAttr().Set(tom_mass)

        physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(tom_prim)
        physx_rb.CreateLinearDampingAttr().Set(2.0)
        physx_rb.CreateAngularDampingAttr().Set(10.0)

        # C. HOJAS / FOLÍOLOS EN LA RAMA DEL PEDICELO (PERMANECEN EN LA PLANTA)
        ped_leaf_positions = [
            (pedicel_mid, 0.65, 35.0),
            (pedicel_mid, 0.65, -35.0),
        ]
        for p_lf_i, (p_lf_pos, p_scale_f, p_side_angle) in enumerate(ped_leaf_positions):
            p_lf_path = f"{pedicels_path}/PedicelLeaflet_{idx+1:02d}_{p_lf_i:02d}"
            p_lf_mesh = create_procedural_leaf_mesh(stage, p_lf_path, p_scale_f, color_leaf)
            p_lf_rot = Gf.Vec3f(random.uniform(-10, 15), p_side_angle, yaw_deg)
            setup_xform(p_lf_mesh, translate=p_lf_pos, rotate_xyz=p_lf_rot)
            
            col_api = UsdPhysics.CollisionAPI.Apply(p_lf_mesh.GetPrim())
            col_api.CreateCollisionEnabledAttr().Set(False)
            mesh_col = UsdPhysics.MeshCollisionAPI.Apply(p_lf_mesh.GetPrim())
            mesh_col.CreateApproximationAttr().Set("convexHull")

        # D. FRUTO DEL TOMATE (Esfera Ovoide PBR)
        fruit_path = f"{t_path}/Fruit"
        fruit_sphere = UsdGeom.Sphere.Define(stage, fruit_path)
        fruit_sphere.CreateRadiusAttr().Set(tom_radius)
        fruit_sphere.CreateDisplayColorAttr().Set([tom_color])

        fruit_center_z = -tom_radius * 1.12 * 0.90
        setup_xform(fruit_sphere, translate=Gf.Vec3d(0.0, 0.0, fruit_center_z), scale=Gf.Vec3f(1.0, 1.0, 1.12))
        UsdPhysics.CollisionAPI.Apply(fruit_sphere.GetPrim())

        physx_mat = UsdPhysics.MaterialAPI.Apply(fruit_sphere.GetPrim())
        physx_mat.CreateStaticFrictionAttr().Set(1.2)
        physx_mat.CreateDynamicFrictionAttr().Set(1.2)

        # D. CÁLIZ / ESTRELLA DE SÉPALOS CURVADOS
        calyx_path = f"{t_path}/Calyx"
        UsdGeom.Xform.Define(stage, calyx_path)

        num_sepals = 6
        sepal_len = tom_radius * 0.70
        for s in range(num_sepals):
            sepal_p = f"{calyx_path}/Sepal_{s:02d}"
            sepal_cube = UsdGeom.Cube.Define(stage, sepal_p)
            sepal_cube.CreateSizeAttr().Set(1.0)
            sepal_cube.CreateDisplayColorAttr().Set([color_calyx])

            angle_deg = s * (360.0 / num_sepals) + random.uniform(-4, 4)
            angle_rad = math.radians(angle_deg)
            
            s_dir_x = math.cos(angle_rad)
            s_dir_y = math.sin(angle_rad)
            
            s_pos = Gf.Vec3d(s_dir_x * (sepal_len / 2.0), s_dir_y * (sepal_len / 2.0), -0.003)
            s_rot = Gf.Vec3f(18.0, 0.0, angle_deg)
            s_scale = Gf.Vec3f(sepal_len, 0.004, 0.0015)

            setup_xform(sepal_cube, translate=s_pos, rotate_xyz=s_rot, scale=s_scale)

        # E. ZONA DE ABSICIÓN (PhysX Breakable Spherical Joint con elasticidad en la punta del pedicelo)
        joint_name = f"TomatoJoint_{idx+1:02d}"
        j_path = f"{joints_path}/{joint_name}"

        joint = UsdPhysics.SphericalJoint.Define(stage, j_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(plant_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(t_path)])

        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(pedicel_end))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(joint.GetPrim())
        filtered_pairs.CreateFilteredPairsRel().SetTargets([Sdf.Path(pedicel_path2), Sdf.Path(t_path)])

        joint.CreateBreakForceAttr().Set(stem_break_force)
        joint.CreateBreakTorqueAttr().Set(stem_break_torque)
        PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())

        tom_drive_api = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        tom_drive_api.CreateStiffnessAttr().Set(10.0)
        tom_drive_api.CreateDampingAttr().Set(15.0)

    print(f"[Éxito] {len(tomato_configs)} tomates independientes generados en la parte alta de la planta.")

    # --------------------------------------------------------------------------
    # 7. INCLUSIÓN DE ROBOTS FRANKA RESEARCH 3 (FR3) (Front y Back Simétricos)
    # --------------------------------------------------------------------------
    if enable_robot:
        franka_usd_url = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd"

        # Robot 1: Front (-X respecto a env_origin)
        r_front_path = f"{world_path}/Franka_Front"
        if stage.GetPrimAtPath(r_front_path).IsValid():
            stage.RemovePrim(r_front_path)

        pos_front = env_origin + Gf.Vec3d(-robot_distance_x, 0.0, 0.0)
        print(f"[Info] Añadiendo Robot Frontal Franka FR3 en {pos_front}...")
        r_front_prim = stage.DefinePrim(r_front_path, "Xform")
        setup_xform(r_front_prim, translate=pos_front, rotate_xyz=Gf.Vec3f(0, 0, 0))
        r_front_prim.GetReferences().AddReference(franka_usd_url)
        UsdPhysics.ArticulationRootAPI.Apply(r_front_prim)

        # Limpiar Robot Back si existía o no es necesario
        r_back_path = f"{world_path}/Franka_Back"
        if stage.GetPrimAtPath(r_back_path).IsValid():
            stage.RemovePrim(r_back_path)

        if num_robots > 1:
            # Robot 2: Back (+X respecto a env_origin)
            pos_back = env_origin + Gf.Vec3d(+robot_distance_x, 0.0, 0.0)
            print(f"[Info] Añadiendo Robot Trasero Franka FR3 en {pos_back}...")
            r_back_prim = stage.DefinePrim(r_back_path, "Xform")
            setup_xform(r_back_prim, translate=pos_back, rotate_xyz=Gf.Vec3f(0, 0, 180))
            r_back_prim.GetReferences().AddReference(franka_usd_url)
            UsdPhysics.ArticulationRootAPI.Apply(r_back_prim)

    # --------------------------------------------------------------------------
    # 8. INCLUSIÓN DE LA CESTA AGRÍCOLA DE RECOLECCIÓN HUECA (Relativa a env_origin)
    # --------------------------------------------------------------------------
    if enable_basket:
        basket_path = f"{world_path}/HarvestBasket"
        abs_basket_pos = env_origin + Gf.Vec3d(basket_pos_x, basket_pos_y, basket_pos_z)
        create_procedural_harvest_basket(stage, basket_path, (abs_basket_pos[0], abs_basket_pos[1], abs_basket_pos[2]))
        print(f"[Éxito] Cesta agrícola de cosecha generada en {basket_path} ({abs_basket_pos})")

    # --------------------------------------------------------------------------
    # 9. CONSOLIDAR Y GUARDAR ESCENA USD
    # --------------------------------------------------------------------------
    if output_usd_path:
        try:
            stage.Export(output_usd_path)
            print(f"[Éxito] Escena de tomatera procedural guardada en: {os.path.abspath(output_usd_path)}")
        except Exception as e:
            print(f"[Aviso] No se pudo exportar directamente a archivo: {e}")

    return stage


if __name__ == "__main__":
    create_procedural_tomato_plant()
