# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 21:17:55 2026
###################################################################
###Evaluación in silico del potencial inhibitorio y modo de unión## 
### de la pristimerina y tingenona frente a la tripanotión       ##    
### reductasa de Trypanosoma cruzi                               ##
################################################################### 
@author: Julio Cesar Henríquez Pérez
"""

import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import math

#Configuración de Parametros  


BASE_DIR = Path("./docking_project")
INPUT_DIR = BASE_DIR / "inputs"
OUTPUT_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

for d in [INPUT_DIR, OUTPUT_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

#Coordenadas del sitio activo de la Tripanotión Reductasa
GRID_BOX = {
    'center_x': 15.520,
    'center_y': 32.140,
    'center_z': -8.230,
    'size_x': 20.0,
    'size_y': 20.0,
    'size_z': 20.0
}

LIGANDS = {
    "pristimerina": "CC1=C(C(=O)C=C2C1=CC=C3[C@]2(CC[C@@]4([C@@]3(CC[C@@]5([C@H]4C[C@](CC5)(C)C(=O)OC)C)C)C)C)O",
    "tingenona": "C[C@@H]1C[C@@H]2[C@@](CC[C@]3([C@]2(CC[C@@]4(C3=CC=C5C4=CC(=O)C(=C5C)O)C)C)C)(CC1=O)C"
}

#Preparación Automatizada del Receptor desde Python 


def preparar_receptor(pdb_origen_path):
    print("receptor para vina")
    receptor_origen = Path(pdb_origen_path)
    receptor_pdbqt = INPUT_DIR / "receptor.pdbqt"
    
    if not receptor_origen.exists():
        raise FileNotFoundError(f"No se encontró el archivo del receptor en: {receptor_origen}")
        
    atomos_copiados = 0
    hetatm_eliminados = 0
    
    with open(receptor_origen, 'r', encoding='utf-8', errors='ignore') as f_in, \
         open(receptor_pdbqt, 'w', encoding='utf-8') as f_out:
        
        for linea in f_in:
            if "HETATM" in linea or "HOH" in linea:
                hetatm_eliminados += 1
                continue
            
            if linea.startswith(("ATOM", "TER", "ANISOU")):
                f_out.write(linea)
                atomos_copiados += 1
                
        f_out.write("END\n")
        
    print(f"Filtrado completado: {atomos_copiados} átomos conservados, {hetatm_eliminados} heteroátomos/ligandos eliminados.")
    print(f"Archivo final listo en: {receptor_pdbqt}")
    return receptor_pdbqt

#Preparación Automatizada de Ligandos

def preparar_ligandos():
    print("Generando conformaciones 3D y ligandos")
    ligand_files = {}
    
    for nombre, smiles in LIGANDS.items():
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        
        res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if res == -1:
            AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
            
        AllChem.MMFFOptimizeMolecule(mol)
        AllChem.ComputeGasteigerCharges(mol)
        
        pdbqt_lines = []
        conf = mol.GetConformer()
        
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            atom_symbol = atom.GetSymbol().upper()
            
            if atom_symbol == 'C':
                ad_type = 'A' if atom.GetIsAromatic() else 'C'
            elif atom_symbol == 'O':
                ad_type = 'OA'
            elif atom_symbol == 'N':
                ad_type = 'NA'
            elif atom_symbol == 'S':
                ad_type = 'SA'
            elif atom_symbol == 'H':
                neighbor = atom.GetNeighbors()[0]
                ad_type = 'HD' if neighbor.GetSymbol() in ['O', 'N'] else 'H'
            else:
                ad_type = atom_symbol
                
            try:
                charge = float(atom.GetProp('_GasteigerCharge'))
            except:
                charge = 0.0
                
            line = f"HETATM{i+1:5d} {atom_symbol:<3s}  UNL A   1    {pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00    {charge:6.3f} {ad_type:>2s}"
            pdbqt_lines.append(line)
            
        pdbqt_lines.insert(0, "ROOT")
        pdbqt_lines.append("ENDROOT")
        pdbqt_lines.append("TORSDOF 0")
        
        pdbqt_path = INPUT_DIR / f"{nombre}.pdbqt"
        with open(pdbqt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(pdbqt_lines) + "\n")
            
        ligand_files[nombre] = pdbqt_path
        print(f"Ligando {nombre} preparado en {pdbqt_path}")
        
    return ligand_files

#Función Auxiliar de Extracción de Energias 


def extraer_energias_vina(log_path):
    energias = []
    with open(log_path, 'r') as f:
        leer = False
        for line in f:
            if "REMARK Vina" in line or "mode |" in line or "-----+------------+------------+------------" in line:
                leer = True
                continue
            if leer:
                parts = line.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    try:
                        energias.append(float(parts[1]))
                    except ValueError:
                        continue
                elif "RMSD" in line or line.strip() == "":
                    if len(energias) > 0 and "Writing output" in line:
                        pass
    if not energias:
        with open(log_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4 and parts[0].isdigit():
                    try:
                        val = float(parts[1])
                        if val < 0:  
                            energias.append(val)
                    except ValueError:
                        continue
    return energias[:9]

#Ejecución de Autodock Vina 

def ejecutar_docking(receptor_pdbqt, ligand_files):
    print("Acoplamiento molecular con AutoDock Vina")
    resultados = []

    vina_exe = BASE_DIR / "vina_1.2.7_win.exe"
    
    if not vina_exe.exists():
        raise FileNotFoundError(f"No se encontró el ejecutable de Vina en: {vina_exe}")

    print(f"Ejecutable de Vina configurado en: {vina_exe}")

    for nombre, lig_path in ligand_files.items():
        output_pdbqt = OUTPUT_DIR / f"{nombre}_out.pdbqt"
        log_path = LOGS_DIR / f"{nombre}_log.txt"
        
        cmd_args = [
            str(vina_exe),
            "--receptor", str(receptor_pdbqt.resolve()),
            "--ligand", str(lig_path.resolve()),
            "--center_x", str(GRID_BOX['center_x']),
            "--center_y", str(GRID_BOX['center_y']),
            "--center_z", str(GRID_BOX['center_z']),
            "--size_x", str(GRID_BOX['size_x']),
            "--size_y", str(GRID_BOX['size_y']),
            "--size_z", str(GRID_BOX['size_z']),
            "--exhaustiveness", "8",
            "--out", str(output_pdbqt.resolve())
        ]
        
        print(f"Ejecutando Acoplamiento para {nombre}...")
        try:
            with open(log_path, 'w') as log_file:
                subprocess.run(cmd_args, stdout=log_file, stderr=subprocess.STDOUT, check=True)
                
            energias = extraer_energias_vina(log_path)
            for i, energy in enumerate(energias, 1):
                resultados.append({
                    'Ligando': nombre,
                    'Pose': i,
                    'Energia_Binding_kcal_mol': energy
                })
            print(f"Docking completado para {nombre}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error Falló el docking para {nombre}.")
            if log_path.exists():
                print("\n--- MENSAJE DE ERROR REGISTRADO EN EL LOG DE VINA ---")
                print(log_path.read_text())
                print("----------------------------------------------------\n")
            else:
                print(f"No se generó archivo de log. Error del sistema: {e}")

    if resultados:
        df_resumen = pd.DataFrame(resultados)
        csv_path = BASE_DIR / "resumen_energias_docking.csv"
        df_resumen.to_csv(csv_path, index=False)
        print(f"Resultados guardados en {csv_path}")
        return df_resumen
    else:
        print("Advertencia No se obtuvieron resultados válidos")
        return pd.DataFrame()


if __name__ == "__main__":
    print("=== PIPELINE INTEGRADO DE BIOINFORMÁTICA ESTRUCTURAL: T. cruzi TR ===")
    
    pdb_origen = Path(r"C:\Users\admin\docking_project\inputs\receptor_chimerax.pdb")
    
    try:
        # Paso 1: Eliminar el receptor PDBQT previo para forzar una regeneración limpia
        receptor_pdbqt_path = INPUT_DIR / "receptor.pdbqt"
        if receptor_pdbqt_path.exists():
            receptor_pdbqt_path.unlink()

        # Paso 2: Preparar receptor
        receptor_preparado = preparar_receptor(pdb_origen)
        
        # Paso 3: Preparar ligandos
        ligandos_preparados = preparar_ligandos()
        
        # Paso 4: Ejecutar AutoDock Vina
        df_resultados = ejecutar_docking(receptor_preparado, ligandos_preparados)
        
        print("\n=== RESUMEN DE ENERGÍAS OBTENIDAS (kcal/mol) ===")
        print(df_resultados)
        
    except Exception as e:
        print(f"\n[ERROR EN EL FLUJO] Ocurrió un problema durante la ejecución: {e}")

#Visualización de Resultados  


def exportar_y_visualizar(df_resultados):
    if df_resultados.empty:
        print("Advertencia El DataFrame está vacío")
        return

    csv_path = OUTPUT_DIR / "resumen_docking_tr_cruzi.csv"
    df_resultados.to_csv(csv_path, index=False)
    print(f"Resultados exportados exitosamente a: {csv_path}")

    plt.figure(figsize=(10, 6))
    
    for ligando in df_resultados["Ligando"].unique():
        df_sub = df_resultados[df_resultados["Ligando"] == ligando]
        plt.plot(
            df_sub["Pose"], 
            df_sub["Energia_Binding_kcal_mol"], 
            marker="o", 
            linewidth=2.5, 
            markersize=8, 
            label=ligando
        )
    
    plt.title("Perfiles de Energía de Unión - T. cruzi Tripanotión Reductasa", fontsize=14, fontweight="bold")
    plt.xlabel("Número de Pose (Conformación)", fontsize=12)
    plt.ylabel("Energía de Unión (kcal/mol)", fontsize=12)
    plt.legend(title="Ligandos Triterpénicos")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "comparacion_energias_docking.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Gráfica guardada exitosamente en: {plot_path}")
    
    plt.show()

exportar_y_visualizar(df_resultados)



#Análisis de Interacciones Moleculares 


def analizar_interacciones_poses(receptor_path, output_dir):
    print("Analizando interacciones moleculares")
    
    atomos_receptor = []
    with open(receptor_path, 'r', encoding='utf-8', errors='ignore') as f:
        for linea in f:
            if linea.startswith("ATOM"):
                try:
                    x = float(linea[30:38])
                    y = float(linea[38:46])
                    z = float(linea[46:54])
                    atom_name = linea[12:16].strip()
                    res_name = linea[17:20].strip()
                    res_seq = linea[22:26].strip()
                    element = atom_name[0].upper()
                    
                    atomos_receptor.append({
                        "x": x, "y": y, "z": z, 
                        "elemento": element, 
                        "residuo": f"{res_name}_{res_seq}",
                        "nombre": atom_name
                    })
                except:
                    continue
                    
    if not atomos_receptor:
        print("Advertencia No se pudieron extraer átomos del receptor.")
        return pd.DataFrame()
        
    interacciones_totales = []
    
    archivos_salida = list(output_dir.glob("*_out.pdbqt")) + list(output_dir.glob("*.pdbqt"))
    
    if not archivos_salida:
        print("Advertencia No se encontraron archivos de poses acoplados en la carpeta outputs.")
        return pd.DataFrame()
        
    for lig_path in archivos_salida:
        nombre_ligando = lig_path.stem.replace("_out", "")
        atomos_ligando = []
        
        with open(lig_path, 'r', encoding='utf-8', errors='ignore') as f:
            for linea in f:
                if linea.startswith(("ATOM", "HETATM")):
                    try:
                        x = float(linea[30:38])
                        y = float(linea[38:46])
                        z = float(linea[46:54])
                        atom_name = linea[12:16].strip()
                        element = atom_name[0].upper()
                        
                        atomos_ligando.append({
                            "x": x, "y": y, "z": z, 
                            "elemento": element,
                            "nombre": atom_name
                        })
                    except:
                        continue
                        
        # Calcular distancias 3D
        for lig_atom in atomos_ligando:
            for rec_atom in atomos_receptor:
                dist = math.sqrt(
                    (lig_atom["x"] - rec_atom["x"])**2 +
                    (lig_atom["y"] - rec_atom["y"])**2 +
                    (lig_atom["z"] - rec_atom["z"])**2
                )
                
                sym_lig = lig_atom["elemento"]
                sym_rec = rec_atom["elemento"]
                
                if dist <= 3.5 and sym_lig in ['O', 'N'] and sym_rec in ['O', 'N']:
                    interacciones_totales.append({
                        "Ligando": nombre_ligando,
                        "Tipo_Interaccion": "Puente de Hidrogeno",
                        "Atom_Ligando": lig_atom["nombre"],
                        "Residuo_Receptor": rec_atom["residuo"],
                        "Atom_Receptor": rec_atom["nombre"],
                        "Distancia_Angstrom": round(dist, 2)
                    })
                
                elif dist <= 4.0 and sym_lig == 'C' and sym_rec == 'C':
                    interacciones_totales.append({
                        "Ligando": nombre_ligando,
                        "Tipo_Interaccion": "Hidrofobica",
                        "Atom_Ligando": lig_atom["nombre"],
                        "Residuo_Receptor": rec_atom["residuo"],
                        "Atom_Receptor": rec_atom["nombre"],
                        "Distancia_Angstrom": round(dist, 2)
                    })

    df_interacciones = pd.DataFrame(interacciones_totales)
    
    if not df_interacciones.empty:
        df_interacciones = df_interacciones.drop_duplicates()
        csv_interacc_path = OUTPUT_DIR / "interacciones_moleculares_detalle.csv"
        df_interacciones.to_csv(csv_interacc_path, index=False)
        print(f"Análisis de interacciones guardado en: {csv_interacc_path}")
        print("\n=== RESUMEN DE INTERACCIONES CLAVE EN SITIO ACTIVO ===")
        print(df_interacciones.head(10))
    else:
        print("No se registraron interacciones en los poses evaluados.")
        
    return df_interacciones

pdb_receptor_limpio = INPUT_DIR / "receptor_vina_limpio.pdb"
df_interacciones = analizar_interacciones_poses(pdb_receptor_limpio, OUTPUT_DIR)


print(df_interacciones["Residuo_Receptor"].value_counts().head(10))


