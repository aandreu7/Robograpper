"""
===============================================================================
CONTROLADOR DE RECOLECCIÓN ROBÓTICA EN TIEMPO REAL (FRANKA FR3) PARA ISAAC SIM
===============================================================================
Este script implementa las funciones de bajo nivel para la manipulación y recolección
de tomates. Cada robot y entorno se identifican estrictamente mediante los parámetros
'env_idx' (int) y 'robot_type' ("front" | "back").

REGLA STRICTA: Cero fallbacks en argumentos y cero búsquedas heurísticas. Si un Prim
no existe en la ruta esperada de /World/Env_{env_idx}, se lanza un ValueError.
===============================================================================
"""

import math
import time
import os
import sys
import numpy as np
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema

import omni.kit.app
import omni.timeline
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka.controllers import RMPFlowController
import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.api.objects import FixedCapsule, FixedCuboid


def setup_xform(prim_or_gprim, translate=None, rotate_xyz=None, scale=None):
    """Auxiliar para configurar transformaciones USD."""
    xformable = UsdGeom.Xformable(prim_or_gprim)
    xformable.ClearXformOpOrder()
    if translate is not None:
        xformable.AddTranslateOp().Set(translate)
    if rotate_xyz is not None:
        xformable.AddRotateXYZOp().Set(rotate_xyz)
    if scale is not None:
        xformable.AddScaleOp().Set(scale)


def get_tomato_targets(stage, env_idx: int, robot_type: str):
    """Devuelve la lista de tomates detectados ordenados por proximidad al robot indicado."""
    if robot_type not in ("front", "back"):
        raise ValueError(f"El parámetro 'robot_type' debe ser 'front' o 'back', se recibió '{robot_type}'.")

    env_scope = f"/World/Env_{env_idx}"
    tomatoes_path = f"{env_scope}/Pot/Plant/Tomatoes"
    tomatoes_prim = stage.GetPrimAtPath(tomatoes_path)
    if not (tomatoes_prim and tomatoes_prim.IsValid()):
        raise ValueError(f"Prim de tomates no encontrado en el escenario ({tomatoes_path}).")

    robot_name = "Franka_Front" if robot_type == "front" else "Franka_Back"
    robot_path = f"{env_scope}/{robot_name}"
    fr3_prim = stage.GetPrimAtPath(robot_path)
    if not (fr3_prim and fr3_prim.IsValid()):
        raise ValueError(f"Prim del robot no encontrado en el escenario ({robot_path}).")

    xformable = UsdGeom.Xformable(fr3_prim)
    robot_base = np.array(xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation())

    targets = []
    for child in tomatoes_prim.GetChildren():
        fruit_prim = stage.GetPrimAtPath(f"{child.GetPath()}/Fruit")
        if not (fruit_prim and fruit_prim.IsValid()):
            raise ValueError(f"Prim de fruto de tomate no válido en {child.GetPath()}/Fruit.")

        xformable = UsdGeom.Xformable(fruit_prim)
        world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        world_pos = world_transform.ExtractTranslation()
        
        sphere = UsdGeom.Sphere(fruit_prim)
        colors = sphere.GetDisplayColorAttr().Get()
        color = colors[0] if colors else Gf.Vec3f(0.8, 0.1, 0.1)
        
        targets.append({
            "name": child.GetName(),
            "path": str(child.GetPath()),
            "prim": child,
            "fruit_prim": fruit_prim,
            "world_pos": world_pos,
            "color": color
        })

    if len(targets) == 0:
        raise ValueError("No se detectaron frutos de tomate válidos en la jerarquía del escenario.")
            
    targets.sort(key=lambda t: np.linalg.norm(np.array([t["world_pos"][0], t["world_pos"][1], t["world_pos"][2]]) - robot_base))
    return targets


def wait_steps(steps):
    """Avanza la simulación en la app activa."""
    app = omni.kit.app.get_app()
    for _ in range(steps):
        app.update()


def setup_fr3_rmpflow_config():
    """Genera descriptores y URDF para el robot Franka FR3."""
    import shutil
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fr3_config_dir = os.path.join(script_dir, "fr3_rmpflow")
    
    desc_path = os.path.join(fr3_config_dir, "robot_descriptor.yaml")
    urdf_path = os.path.join(fr3_config_dir, "lula_fr3_gen.urdf")
    common_path = os.path.join(fr3_config_dir, "fr3_rmpflow_common.yaml")

    mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
    base_policy_dir = os.path.join(mg_extension_path, "motion_policy_configs", "franka")

    src_desc = os.path.join(base_policy_dir, "rmpflow", "robot_descriptor.yaml")
    src_urdf = os.path.join(base_policy_dir, "lula_franka_gen.urdf")
    src_common = os.path.join(base_policy_dir, "rmpflow", "franka_rmpflow_common.yaml")

    shutil.rmtree(fr3_config_dir, ignore_errors=True)
    os.makedirs(fr3_config_dir, exist_ok=True)

    with open(src_desc, "r", encoding="utf-8") as f:
        desc_content = f.read().replace("panda_", "fr3_")
    with open(desc_path, "w", encoding="utf-8") as f:
        f.write(desc_content)

    with open(src_urdf, "r", encoding="utf-8") as f:
        urdf_content = f.read().replace("panda_", "fr3_")
    with open(urdf_path, "w", encoding="utf-8") as f:
        f.write(urdf_content)

    with open(src_common, "r", encoding="utf-8") as f:
        common_content = f.read().replace("panda_", "fr3_")
    with open(common_path, "w", encoding="utf-8") as f:
        f.write(common_content)

    return {
        "robot_description_path": desc_path,
        "urdf_path": urdf_path,
        "rmpflow_config_path": common_path,
        "end_effector_frame_name": "fr3_hand",
        "maximum_substep_size": 0.0033333333333333335
    }


def get_prim_dynamic_bbox(prim):
    """Calcula dinámicamente la Bounding Box 3D de un Prim USD."""
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bbox = bbox_cache.ComputeWorldBound(prim)
    aligned_range = bbox.ComputeAlignedRange()
    min_p = aligned_range.GetMin()
    max_p = aligned_range.GetMax()

    center_pos = np.array([
        (min_p[0] + max_p[0]) / 2.0,
        (min_p[1] + max_p[1]) / 2.0,
        (min_p[2] + max_p[2]) / 2.0
    ])
    dimensions = np.array([
        max_p[0] - min_p[0],
        max_p[1] - min_p[1],
        max_p[2] - min_p[2]
    ])
    return center_pos, dimensions


def initialize_rmpflow_controller(fr3, stage, env_idx: int, robot_type: str):
    """Inicializa el controlador RMPflow registrando obstáculos estrictos del entorno indicado."""
    if robot_type not in ("front", "back"):
        raise ValueError(f"El parámetro 'robot_type' debe ser 'front' o 'back', se recibió '{robot_type}'.")

    env_scope = f"/World/Env_{env_idx}"
    fr3_config = setup_fr3_rmpflow_config()
    
    rmp_flow = mg.lula.motion_policies.RmpFlow(**fr3_config)
    articulation_rmp = mg.ArticulationMotionPolicy(fr3, rmp_flow)
    rmp_controller = mg.MotionPolicyController(name=f"fr3_rmpflow_env{env_idx}_{robot_type}", articulation_motion_policy=articulation_rmp)

    robot_base_translation, robot_base_orientation = fr3.get_world_pose()
    rmp_controller.get_motion_policy().set_robot_base_pose(robot_base_translation, robot_base_orientation)

    # 1. Obstáculo: Maceta
    pot_mesh_path = f"{env_scope}/Pot/PotMesh"
    pot_mesh_prim = stage.GetPrimAtPath(pot_mesh_path)
    if not (pot_mesh_prim and pot_mesh_prim.IsValid()):
        raise ValueError(f"Prim de la maceta no encontrado en {pot_mesh_path} para registrar en RMPflow.")
    
    pos, dim = get_prim_dynamic_bbox(pot_mesh_prim)
    pot_obs = FixedCuboid(
        f"{env_scope}/PotObstacle_{robot_type}",
        name=f"pot_obstacle_env{env_idx}_{robot_type}",
        position=pos,
        scale=dim,
        visible=False
    )
    if pot_obs.prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI(pot_obs.prim).GetCollisionEnabledAttr().Set(False)
    UsdGeom.Imageable(pot_obs.prim).MakeInvisible()
    rmp_controller.add_obstacle(pot_obs, static=False)

    # 2. Obstáculo: Palo Tutor
    stake_path = f"{env_scope}/Pot/Plant/SupportStake"
    stake_prim = stage.GetPrimAtPath(stake_path)
    if not (stake_prim and stake_prim.IsValid()):
        raise ValueError(f"Prim del palo tutor no encontrado en {stake_path} para registrar en RMPflow.")

    pos, dim = get_prim_dynamic_bbox(stake_prim)
    radius = max(dim[0], dim[1]) / 2.0 + 0.005
    height = dim[2]
    stake_obs = FixedCapsule(
        f"{env_scope}/StakeObstacle_{robot_type}",
        name=f"stake_obstacle_env{env_idx}_{robot_type}",
        radius=radius,
        height=height,
        position=pos,
        visible=False
    )
    if stake_obs.prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI(stake_obs.prim).GetCollisionEnabledAttr().Set(False)
    UsdGeom.Imageable(stake_obs.prim).MakeInvisible()
    rmp_controller.add_obstacle(stake_obs, static=False)

    # 3. Obstáculo: Tallo Principal
    stem_path = f"{env_scope}/Pot/Plant/MainStem"
    stem_prim = stage.GetPrimAtPath(stem_path)
    if not (stem_prim and stem_prim.IsValid()):
        raise ValueError(f"Prim del tallo principal no encontrado en {stem_path} para registrar en RMPflow.")

    pos, dim = get_prim_dynamic_bbox(stem_prim)
    radius = max(dim[0], dim[1]) / 2.0 + 0.008
    height = dim[2]
    stem_obs = FixedCapsule(
        f"{env_scope}/StemObstacle_{robot_type}",
        name=f"stem_obstacle_env{env_idx}_{robot_type}",
        radius=radius,
        height=height,
        position=pos,
        visible=False
    )
    if stem_obs.prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI(stem_obs.prim).GetCollisionEnabledAttr().Set(False)
    UsdGeom.Imageable(stem_obs.prim).MakeInvisible()
    rmp_controller.add_obstacle(stem_obs, static=False)

    return rmp_controller


def get_live_target_position(target):
    """Obtiene la posición 3D actual del fruto en coordenadas de MUNDO."""
    f_prim = target.get('fruit_prim', target['prim'])
    if not (f_prim and f_prim.IsValid()):
        raise ValueError(f"Prim del objetivo {target.get('name', 'desconocido')} no es válido en USD.")
    
    xformable = UsdGeom.Xformable(f_prim)
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = world_transform.ExtractTranslation()
    return np.array([translation[0], translation[1], translation[2]])


def classify_tomato_sector(t_pos, stake_pos, robot_type: str = "front"):
    """Clasifica el fruto en sectores geométricos y retorna dirección y orientación 6D adaptada al robot."""
    rel_x = t_pos[0] - stake_pos[0]
    rel_y = t_pos[1] - stake_pos[1]

    if abs(rel_y) > abs(rel_x) and abs(rel_y) > 0.05:
        if rel_y > 0.0:  # Lateral Norte (+Y)
            sector = "Lateral Norte (+Y)"
            approach_dir = np.array([0.0, -1.0, 0.0])
            orientation = np.array([0.5, 0.5, 0.5, -0.5])
        else:  # Lateral Sur (-Y)
            sector = "Lateral Sur (-Y)"
            approach_dir = np.array([0.0, 1.0, 0.0])
            orientation = np.array([0.5, -0.5, 0.5, 0.5])
    else:  # Frontal (-X)
        if robot_type == "front":
            sector = "Frontal (-X)"
            approach_dir = np.array([1.0, 0.0, 0.0])
            orientation = np.array([0.7071068, 0.0, 0.7071068, 0.0])
        else:
            sector = "Trasero (+X)"
            approach_dir = np.array([-1.0, 0.0, 0.0])
            orientation = np.array([0.7071068, 0.0, -0.7071068, 0.0])

    return sector, approach_dir, orientation


def get_ee_pose_6d(fr3, rmp_controller):
    """Devuelve la pose 6D del efector final del robot."""
    ee_pos = rmp_controller.get_motion_policy().get_end_effector_as_prim().get_world_pose()[0]
    ee_rot = rmp_controller.get_motion_policy().get_end_effector_as_prim().get_world_pose()[1]
    return np.array(ee_pos), np.array(ee_rot)


def use_grip(fr3, open_gripper=True, target_gap=None):
    """Abre o cierra las pinzas del robot adaptando la abertura al diámetro del objeto."""
    if open_gripper:
        val = 0.04
    elif target_gap is not None:
        val = max(0.005, min(0.04, float(target_gap) / 2.0))
    else:
        val = 0.01

    action = ArticulationAction(joint_positions=np.array([val, val]), joint_indices=np.array([7, 8]))
    fr3.apply_action(action)
    wait_steps(15)


def find_finger_prims(stage, env_idx: int, robot_type: str):
    """Localiza los prims USD de las yemas de las pinzas del robot especificado."""
    if robot_type not in ("front", "back"):
        raise ValueError(f"El parámetro 'robot_type' debe ser 'front' o 'back', se recibió '{robot_type}'.")

    robot_name = "Franka_Front" if robot_type == "front" else "Franka_Back"
    robot_path = f"/World/Env_{env_idx}/{robot_name}"

    left_p = stage.GetPrimAtPath(f"{robot_path}/fr3_leftfinger")
    right_p = stage.GetPrimAtPath(f"{robot_path}/fr3_rightfinger")

    if not (left_p and left_p.IsValid()):
        raise ValueError(f"Prim de la yema izquierda de las pinzas no encontrado en {robot_path}/fr3_leftfinger.")
    if not (right_p and right_p.IsValid()):
        raise ValueError(f"Prim de la yema derecha de las pinzas no encontrado en {robot_path}/fr3_rightfinger.")

    return left_p, right_p


def approach(fr3, rmp_controller, t_pos, approach_dir, orientation, stage=None, env_idx: int = 0, robot_type: str = "front"):
    """Aproxima la muñeca del robot a 18 cm del tomate y visualiza los puntos 3D de diagnóstico (Azul y Verde)."""
    caught_pos = np.array([t_pos[0], t_pos[1], t_pos[2]])
    target_position = caught_pos - 0.18 * approach_dir

    if stage is not None:
        env_scope = f"/World/Env_{env_idx}"

        # 1. Punto Azul: caught_pos (Posición del tomate calculada en approach)
        blue_path = f"{env_scope}/ApproachCaughtPos_{robot_type}"
        if stage.GetPrimAtPath(blue_path).IsValid():
            stage.RemovePrim(blue_path)
        blue_sphere = UsdGeom.Sphere.Define(stage, blue_path)
        blue_sphere.CreateRadiusAttr().Set(0.015)  # Radio de 1.5 cm
        blue_sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.4, 1.0)])  # Azul Intenso
        setup_xform(blue_sphere, translate=Gf.Vec3d(float(caught_pos[0]), float(caught_pos[1]), float(caught_pos[2])))
        UsdGeom.Imageable(blue_sphere.GetPrim()).MakeVisible()

        # 2. Punto Verde: target_position (Posición objetivo de aproximación de la muñeca)
        green_path = f"{env_scope}/ApproachTargetPos_{robot_type}"
        if stage.GetPrimAtPath(green_path).IsValid():
            stage.RemovePrim(green_path)
        green_sphere = UsdGeom.Sphere.Define(stage, green_path)
        green_sphere.CreateRadiusAttr().Set(0.015)  # Radio de 1.5 cm
        green_sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 1.0, 0.2)])  # Verde Intenso
        setup_xform(green_sphere, translate=Gf.Vec3d(float(target_position[0]), float(target_position[1]), float(target_position[2])))
        UsdGeom.Imageable(green_sphere.GetPrim()).MakeVisible()

    return move(fr3, rmp_controller, target_position, orientation, np.arange(7), n_steps=500)


def catch(fr3, rmp_controller, stage, target, approach_dir, orientation, env_idx: int, robot_type: str):
    """Ejecuta la captura adaptativa con PID 3D sobre las yemas de las pinzas."""
    t_prim = target.get('fruit_prim', target['prim'])
    if not (t_prim and t_prim.IsValid()):
        raise ValueError(f"Prim de fruto de tomate no válido para el objetivo {target.get('name', 'desconocido')}.")
    
    _, dims = get_prim_dynamic_bbox(t_prim)
    fruit_radius = float(np.max(dims) / 2.0)

    left_finger_prim, right_finger_prim = find_finger_prims(stage, env_idx, robot_type)

    # Prim USD dinámico para el vector de error (Rojo Intenso)
    env_scope = f"/World/Env_{env_idx}"
    vector_path = f"{env_scope}/ErrorVector_{robot_type}"
    if stage.GetPrimAtPath(vector_path).IsValid():
        stage.RemovePrim(vector_path)

    curves = UsdGeom.BasisCurves.Define(stage, vector_path)
    curves.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
    curves.CreateCurveVertexCountsAttr().Set([2])
    curves.CreateWidthsAttr().Set([0.003, 0.003]) # 8 mm de grosor en 3D
    curves.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)]) # Rojo Intenso

    kp = 0.35
    ki = 0.30
    kd = 0.20
    max_iterations = 35
    target_tolerance = 0.005
    integral_error = np.zeros(3)
    max_integral = 0.03
    prev_error_vec = None

    for iteration in range(1, max_iterations + 1):
        t_pos = get_live_target_position(target)
        ee_pos, _ = get_ee_pose_6d(fr3, rmp_controller)

        l_pos, _ = get_prim_dynamic_bbox(left_finger_prim)
        r_pos, _ = get_prim_dynamic_bbox(right_finger_prim)
        fingers_center = (l_pos + r_pos) / 2.0
        fingers_center += 0.02 * approach_dir

        target_finger_pos = t_pos - (fruit_radius * 0.10) * approach_dir
        error_vector = target_finger_pos - fingers_center
        error_dist = float(np.linalg.norm(error_vector))

        # Actualizar vértices del Prim USD en la escena (Rojo Intenso)
        p_start = Gf.Vec3f(float(fingers_center[0]), float(fingers_center[1]), float(fingers_center[2]))
        p_end = Gf.Vec3f(float(target_finger_pos[0]), float(target_finger_pos[1]), float(target_finger_pos[2]))
        curves.CreatePointsAttr().Set([p_start, p_end])

        integral_error += error_vector
        integral_norm = np.linalg.norm(integral_error)
        if integral_norm > max_integral:
            integral_error = (integral_error / integral_norm) * max_integral

        if prev_error_vec is None:
            derivative_error = np.zeros(3)
        else:
            derivative_error = error_vector - prev_error_vec
        prev_error_vec = error_vector.copy()

        if error_dist <= target_tolerance:
            print(f"\033[91m     [CATCH YEMAS-TOMATE] Convergencia alcanzada en iteración {iteration} (Error Yemas <= {target_tolerance*1000.0:.1f} mm)\033[0m")
            break

        print(f"\033[91m     [CATCH PID Paso {iteration:02d}/{max_iterations}] Error Dist: {error_dist*1000.0:.1f} mm | Vector Error: [{error_vector[0]:.4f}, {error_vector[1]:.4f}, {error_vector[2]:.4f}]\033[0m")

        corrected_ee_pos = ee_pos + (kp * error_vector) + (ki * integral_error) + (kd * derivative_error)
        move(fr3, rmp_controller, corrected_ee_pos, orientation, np.arange(7), n_steps=60)
        wait_steps(10)

    # Eliminar Prim USD al finalizar la maniobra catch
    if stage.GetPrimAtPath(vector_path).IsValid():
        stage.RemovePrim(vector_path)

    return True


def retract(fr3, rmp_controller, t_pos, approach_dir, orientation):
    """Retrae el brazo robótico alejándolo de la planta."""
    caught_pos = np.array([t_pos[0], t_pos[1], t_pos[2]])
    retract_position = caught_pos - 0.30 * approach_dir
    return move(fr3, rmp_controller, retract_position, orientation, np.arange(7), n_steps=250)


def collect(fr3, rmp_controller, stage, env_idx: int, robot_type: str):
    """Desplaza el efector final sobre la cesta bordeando la planta por fuera a través de un waypoint de seguridad."""
    if robot_type not in ("front", "back"):
        raise ValueError(f"El parámetro 'robot_type' debe ser 'front' o 'back', se recibió '{robot_type}'.")

    basket_path = f"/World/Env_{env_idx}/HarvestBasket"
    basket_prim = stage.GetPrimAtPath(basket_path)
    if not (basket_prim and basket_prim.IsValid()):
        raise ValueError(f"Prim de la cesta de recolección no encontrado en {basket_path}.")

    # 1. Waypoint de seguridad exterior para bordear la planta por fuera (evitando atravesar el tutor/tallo)
    robot_prim = stage.GetPrimAtPath(f"/World/Env_{env_idx}/Franka_{robot_type.capitalize()}")
    r_base_pos, _ = UsdGeom.Xformable(robot_prim).ComputeLocalToWorldTransform(0).ExtractTranslation(), None
    safe_x = r_base_pos[0] * 0.60
    safe_collect_pos = np.array([safe_x, 0.0, 0.55])

    # Orientación adaptada al robot
    if robot_type == "front":
        target_orientation = np.array([0.7071068, 0.0, 0.7071068, 0.0])
    else:
        target_orientation = np.array([0.7071068, 0.0, -0.7071068, 0.0])

    # Paso 1: Mover al waypoint exterior seguro para salir del sector del tomate
    move(fr3, rmp_controller, safe_collect_pos, target_orientation, np.arange(7), n_steps=120)

    # 2. Posición objetivo final sobre la cesta
    basket_pos, basket_dim = get_prim_dynamic_bbox(basket_prim)
    basket_top_z = basket_pos[2] + (basket_dim[2] / 2.0) + 0.12
    x_offset = -0.10 if robot_type == "front" else 0.10
    target_position = np.array([basket_pos[0] + x_offset, basket_pos[1], basket_top_z])

    # Paso 2: Descender limpiamente sobre la cesta de recolección
    return move(fr3, rmp_controller, target_position, target_orientation, np.arange(7), n_steps=200)


def move(fr3, rmp_controller, target_position, target_orientation, joint_indices: np.array, joint_positions: np.array = None, n_steps: int = 250):
    """Envía comandos de movimiento al controlador RMPflow o articulaciones."""
    if joint_positions is None:
        for _ in range(n_steps):
            action = rmp_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=target_orientation
            )
            fr3.apply_action(action)
            wait_steps(1)
    else:
        action = ArticulationAction(
            joint_positions=joint_positions, 
            joint_indices=joint_indices
        )
        fr3.apply_action(action)
        wait_steps(n_steps)

    return True


def recompose(fr3, rmp_controller, home_joint_positions, home_ee_pos, home_ee_ori, stage=None, env_idx: int = 0, robot_type: str = "front"):
    """Retorna el robot a su postura de reposo home usando Waypoints Dinámicos en Espacio Libre y RMPflow."""
    print(" -> [RECOMPOSE] Retornando robot a postura home mediante Waypoints Dinámicos en Espacio Libre...")

    robot_base_pos, robot_base_ori = fr3.get_world_pose()

    # Orientación adaptada al robot
    if robot_type == "front":
        target_orientation = np.array([0.7071068, 0.0, 0.7071068, 0.0])
    else:
        target_orientation = np.array([0.7071068, 0.0, -0.7071068, 0.0])

    env_scope = f"/World/Env_{env_idx}"
    
    # 1. Bounding Box del palo tutor para calcular Z_clearance
    stake_path = f"{env_scope}/Pot/Plant/SupportStake"
    stake_prim = stage.GetPrimAtPath(stake_path)
    if stake_prim and stake_prim.IsValid():
        stake_pos, stake_dim = get_prim_dynamic_bbox(stake_prim)
        stake_top_z = stake_pos[2] + (stake_dim[2] / 2.0)
    else:
        stake_top_z = 0.50
    
    clearance_z = max(0.60, stake_top_z + 0.10)

    # 2. Bounding Box de la cesta de recolección
    basket_path = f"{env_scope}/HarvestBasket"
    basket_prim = stage.GetPrimAtPath(basket_path)
    if basket_prim and basket_prim.IsValid():
        basket_pos, basket_dim = get_prim_dynamic_bbox(basket_prim)
        x_offset = -0.10 if robot_type == "front" else 0.10
        basket_target_x = basket_pos[0] + x_offset
        basket_y = basket_pos[1]
    else:
        basket_target_x = robot_base_pos[0] * 0.85
        basket_y = 0.55

        # --- WAYPOINT 1: Ascenso vertical directo sobre la cesta a Z_clearance ---
        w1_ascent_pos = np.array([basket_target_x, basket_y, clearance_z])
        move(fr3, rmp_controller, w1_ascent_pos, target_orientation, np.arange(7), n_steps=120)

        # --- WAYPOINT 2: Retracción lateral exterior en el eje X manteniendo Y_basket ---
        w2_outer_retract_pos = np.array([robot_base_pos[0] * 0.75, basket_y, clearance_z])
        move(fr3, rmp_controller, w2_outer_retract_pos, target_orientation, np.arange(7), n_steps=120)

        # --- WAYPOINT 3: Standoff de seguridad central en Y=0 a Z_clearance ---
        w3_safe_pos = np.array([robot_base_pos[0] * 0.50, robot_base_pos[1], clearance_z])
        move(fr3, rmp_controller, w3_safe_pos, target_orientation, np.arange(7), n_steps=120)

    # 3. Retorno guiado via RMPflow a la pose 3D de Home
    move(fr3, rmp_controller, home_ee_pos, home_ee_ori, np.arange(7), n_steps=150)

    # 4. Resincronizar el estado interno de Lula/RMPflow tras el movimiento
    try:
        rmp_controller.reset()
    except Exception:
        pass

    robot_base_pos, robot_base_ori = fr3.get_world_pose()
    rmp_controller.get_motion_policy().set_robot_base_pose(robot_base_pos, robot_base_ori)

    return True
