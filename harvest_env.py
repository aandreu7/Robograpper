"""
===============================================================================
CLASE DE ENTORNO ENCAPSULADO CON ORIGEN PROPIO Y DUAL-ROBOT SIMÉTRICO
===============================================================================
Este script encapsula la clase 'TomatoHarvestEnv' integrada con SimulationApp.
Cada entorno posee un origen de coordenadas 3D propio ('env_origin') configurable
manualmente en el constructor, anclado al centro de la planta de tomate.

REGLA STRICTA: Sin fallbacks. Los robots y elementos se identifican mediante
el índice de entorno 'env_idx' y el tipo de robot 'robot_type' ("front" | "back").

INSTRUCCIONES DE EJECUCIÓN:
  python harvest_env.py
===============================================================================
"""

import os
import sys
import time
import numpy as np

# Configurar redirección de salida a archivo de log en IsaacSim_scripts
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except Exception:
    script_dir = r"c:\Users\andre\Desktop\IsaacSim\IsaacSim_scripts"

log_file_path = os.path.join(script_dir, "harvest_log.txt")

class LoggerTee:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log_file = open(filepath, "w", encoding="utf-8")
        self.log_file.write(f"=== INICIO LOG DE COSECHA ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===\n")
        self.log_file.flush()

    def write(self, message):
        if self.terminal:
            try:
                self.terminal.write(message)
            except Exception:
                pass
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        if self.terminal:
            try:
                self.terminal.flush()
            except Exception:
                pass
        self.log_file.flush()

sys.stdout = LoggerTee(log_file_path)
sys.stderr = sys.stdout

import omni.kit.app

simulation_app = None

# Detectar si estamos dentro del IDE/GUI activo de Isaac Sim o ejecutando desde terminal externa
if omni.kit.app.get_app() is None or not omni.kit.app.get_app().is_running():
    try:
        from isaacsim import SimulationApp
        simulation_app = SimulationApp({"headless": False})
    except Exception as e:
        print(f"[Aviso] No se pudo instanciar SimulationApp: {e}")
        simulation_app = None

import omni.usd
import omni.timeline
from pxr import Gf, UsdGeom

import importlib.util

def _load_script_module(module_name, file_basename):
    if module_name in sys.modules:
        del sys.modules[module_name]

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        script_dir = r"c:\Users\andre\Desktop\IsaacSim\IsaacSim_scripts"
    
    file_path = os.path.join(script_dir, file_basename)
    if not os.path.exists(file_path):
        file_path = os.path.join(r"c:\Users\andre\Desktop\IsaacSim\IsaacSim_scripts", file_basename)
        
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

generate_tomato_plant = _load_script_module("generate_tomato_plant", "generate_tomato_plant.py")
harvest_tomatoes = _load_script_module("harvest_tomatoes", "harvest_tomatoes.py")

from isaacsim.core.prims import SingleArticulation as Articulation

create_procedural_tomato_plant = generate_tomato_plant.create_procedural_tomato_plant
initialize_rmpflow_controller = harvest_tomatoes.initialize_rmpflow_controller
get_tomato_targets = harvest_tomatoes.get_tomato_targets
get_live_target_position = harvest_tomatoes.get_live_target_position
approach = harvest_tomatoes.approach
catch = harvest_tomatoes.catch
retract = harvest_tomatoes.retract
collect = harvest_tomatoes.collect
recompose = harvest_tomatoes.recompose
use_grip = harvest_tomatoes.use_grip
get_prim_dynamic_bbox = harvest_tomatoes.get_prim_dynamic_bbox
classify_tomato_sector = harvest_tomatoes.classify_tomato_sector


class TomatoHarvestEnv:
    """
    Entorno modular encapsulado con origen de coordenadas propio (env_origin) y
    soporte para hasta 2 robots Franka FR3 simétricos con orientación adaptativa.
    """

    def __init__(self, env_origin=np.array([0.0, 0.0, 0.0]), num_robots=1, env_idx=0):
        if num_robots not in (1, 2):
            raise ValueError("El parámetro 'num_robots' debe ser 1 o 2.")

        self.env_origin = np.array(env_origin, dtype=float)
        self.num_robots = num_robots
        self.env_idx = env_idx
        self.env_scope_path = f"/World/Env_{env_idx}"

        self.simulation_app = simulation_app
        self.timeline = None
        self.stage = None
        self.robots = {}
        self.stake_pos = None

    def reset(self):
        """Genera el escenario en env_origin con robots simétricos y particiona los frutos sin solapamiento."""
        print(f"[TomatoHarvestEnv {self.env_idx}] Generando escenario en origen {self.env_origin} ({self.num_robots} robots)...")
        gf_origin = Gf.Vec3d(float(self.env_origin[0]), float(self.env_origin[1]), float(self.env_origin[2]))
        self.stage = generate_tomato_plant.create_procedural_tomato_plant(env_origin=gf_origin, env_idx=self.env_idx, num_robots=self.num_robots)

        self.timeline = omni.timeline.get_timeline_interface()
        self.timeline.play()

        # Asentar físicas
        self.step(20)

        stake_path = f"{self.env_scope_path}/Pot/Plant/SupportStake"
        stake_prim = self.stage.GetPrimAtPath(stake_path)
        if not (stake_prim and stake_prim.IsValid()):
            raise ValueError(f"Prim del palo tutor no encontrado en {stake_path}.")
        self.stake_pos, _ = get_prim_dynamic_bbox(stake_prim)

        # Detectar todos los tomates del entorno
        all_tomatoes = get_tomato_targets(self.stage, env_idx=self.env_idx, robot_type="front")

        # 1. Configurar Robot Frontal
        self.robots = {}
        r_front_path = f"{self.env_scope_path}/Franka_Front"
        if not self.stage.GetPrimAtPath(r_front_path).IsValid():
            raise ValueError(f"Prim de robot frontal no encontrado en {r_front_path}.")

        fr3_front = Articulation(r_front_path)
        fr3_front.initialize()
        rmp_front = initialize_rmpflow_controller(fr3_front, self.stage, env_idx=self.env_idx, robot_type="front")

        self.robots["front"] = {
            "name": "Robot_Frontal",
            "key": "front",
            "articulation": fr3_front,
            "controller": rmp_front,
            "path": r_front_path,
            "targets": []
        }

        # 2. Configurar Robot Trasero si num_robots == 2
        if self.num_robots > 1:
            r_back_path = f"{self.env_scope_path}/Franka_Back"
            if not self.stage.GetPrimAtPath(r_back_path).IsValid():
                raise ValueError(f"Prim de robot trasero no encontrado en {r_back_path}.")

            fr3_back = Articulation(r_back_path)
            fr3_back.initialize()
            rmp_back = initialize_rmpflow_controller(fr3_back, self.stage, env_idx=self.env_idx, robot_type="back")

            self.robots["back"] = {
                "name": "Robot_Trasero",
                "key": "back",
                "articulation": fr3_back,
                "controller": rmp_back,
                "path": r_back_path,
                "targets": []
            }

        # Partición de frutos sin solapamiento
        self._partition_targets(all_tomatoes)

        self.step(20)
        return {r_key: len(r_info["targets"]) for r_key, r_info in self.robots.items()}

    def _partition_targets(self, all_tomatoes):
        """Particiona los frutos sin solapamiento entre los robots disponibles del entorno."""
        for t in all_tomatoes:
            t_pos = get_live_target_position(t)
            rel_x = t_pos[0] - self.stake_pos[0]

            if self.num_robots == 1:
                if rel_x <= 0.0:
                    self.robots["front"]["targets"].append(t)
            else:
                if rel_x < 0.0:
                    self.robots["front"]["targets"].append(t)
                else:
                    self.robots["back"]["targets"].append(t)

        print(f"[TomatoHarvestEnv {self.env_idx}] Partición de frutos (Sin Solapamiento):")
        for r_key, r_info in self.robots.items():
            print(f"  - {r_info['name']}: {len(r_info['targets'])} tomates asignados.")

    def _get_robot_approach_config(self, r_key, t_pos):
        """Calcula el sector geométrico, el vector de aproximación y la orientación 6D utilizando classify_tomato_sector."""
        return classify_tomato_sector(t_pos, self.stake_pos, robot_type=r_key)

    def step(self, steps=1):
        """Avanza la simulación sincronizando en modo standalone o GUI."""
        if self.simulation_app is not None:
            for _ in range(steps):
                self.simulation_app.step(render=True)
        else:
            app = omni.kit.app.get_app()
            for _ in range(steps):
                app.update()

    def _harvest_robot_targets(self, r_info):
        """Ejecuta el ciclo de cosecha de un robot adaptando su tipo y entorno estrictos."""
        fr3 = r_info["articulation"]
        rmp_controller = r_info["controller"]
        targets = r_info["targets"]
        robot_name = r_info["name"]
        r_key = r_info["key"]

        # if len(targets) > 2:
        #     print(f"[{robot_name}] Omitiendo los 2 primeros tomates ({targets[0]['name']}, {targets[1]['name']}) por debug...")
        #     targets = targets[2:]

        if not targets:
            print(f"[{robot_name}] No tiene tomates asignados en su sector.")
            return

        home_joint_positions = fr3.get_joint_positions().copy()
        home_ee_pos, home_ee_ori = harvest_tomatoes.get_ee_pose_6d(fr3, rmp_controller)

        for idx, target in enumerate(targets):
            t_name = target['name']
            t_pos = get_live_target_position(target)
            
            sector, app_dir, orientation = self._get_robot_approach_config(r_key, t_pos)

            print(f"\n---> [{robot_name}] Cosechando Tomate {idx+1}/{len(targets)}: {t_name} | Sector Clasificado: {sector}...")

            # 1. APPROACH
            t_pos = get_live_target_position(target)
            approach(fr3, rmp_controller, t_pos, app_dir, orientation, stage=self.stage, env_idx=self.env_idx, robot_type=r_key)
            use_grip(fr3, True)

            # 2. CATCH (PID Adaptativo 3D con env_idx y robot_type)
            catch(fr3, rmp_controller, self.stage, target, app_dir, orientation, env_idx=self.env_idx, robot_type=r_key)
            t_prim = target.get('fruit_prim', target['prim'])
            _, dims = get_prim_dynamic_bbox(t_prim)
            fruit_radius = float(np.max(dims) / 2.0)
            use_grip(fr3, False, target_gap=fruit_radius * 1.6)

            # 3. RETRACT
            t_pos = get_live_target_position(target)
            retract(fr3, rmp_controller, t_pos, app_dir, orientation)

            # 4. COLLECT (Cesta estricta de este entorno, adaptada al robot)
            collect(fr3, rmp_controller, self.stage, env_idx=self.env_idx, robot_type=r_key)
            use_grip(fr3, True)
            self.step(20)

            # 5. RECOMPOSE (Retorno guiado via Waypoints Dinámicos en Espacio Libre + RMPflow)
            recompose(fr3, rmp_controller, home_joint_positions, home_ee_pos=home_ee_pos, home_ee_ori=home_ee_ori, stage=self.stage, env_idx=self.env_idx, robot_type=r_key)

    def harvest_all(self):
        """Ejecuta la recolección coordinada de todos los robots de este entorno."""
        if not self.robots:
            self.reset()

        for r_key, r_info in self.robots.items():
            self._harvest_robot_targets(r_info)

        print(f"\n[Éxito] Recolección completada por todos los robots del entorno {self.env_idx}.")

    def close(self):
        """Cierra la aplicación si fue lanzada externamente."""
        if self.simulation_app is not None:
            print("[TomatoHarvestEnv] Cerrando SimulationApp...")
            self.simulation_app.close()
        else:
            print("[TomatoHarvestEnv] Ejecución en GUI completada.")


def main():
    env = TomatoHarvestEnv(env_origin=np.array([0.0, 0.0, 0.0]), num_robots=1, env_idx=0)
    env.reset()
    env.harvest_all()
    env.step(50)
    env.close()


main()
