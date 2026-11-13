"""
================================================================================
Documentación Técnica: Segmentación Óptima de Imágenes Botánicas mediante
Espacios Perceptuales y Morfología Matemática
================================================================================
Institución : Universidad Politécnica Metropolitana de Hidalgo (UPMH)
Programa    : Maestría en Inteligencia Artificial
Asignatura  : Sistemas de Visión Artificial
Docente     : Dr. Ernesto García Amaro
Alumno      : Ing. Eduardo Taurino Martínez Morales
Archivo     : segmentacion_hojas_secuencia.py
================================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ==============================================================================
# 1. FUNCIONES DE LECTURA Y ESCRITURA SEGURAS (COMPATIBLES CON TILDES EN WINDOWS)
# ==============================================================================
def cargar_imagen_segura(ruta: str):
    if os.path.exists(ruta):
        bytes_img = np.fromfile(ruta, dtype=np.uint8)
        return cv2.imdecode(bytes_img, cv2.IMREAD_COLOR)
    return None


def guardar_imagen_segura(ruta: str, img: np.ndarray) -> None:
    ext = os.path.splitext(ruta)[1]
    exito, buffer = cv2.imencode(ext, img)
    if exito:
        buffer.tofile(ruta)


# ==============================================================================
# 2. REFINAMIENTO MORFOLÓGICO Y EXTRACCIÓN DE COMPONENTE PRINCIPAL
# ==============================================================================
def refinar_mascara(mascara_cruda: np.ndarray):
    kernel_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_apertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    m = cv2.morphologyEx(mascara_cruda, cv2.MORPH_CLOSE, kernel_cierre, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel_apertura, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    mascara_limpia = np.zeros_like(m)
    if num_labels > 1:
        etiqueta_mayor = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mascara_limpia[labels == etiqueta_mayor] = 255

    contornos, _ = cv2.findContours(
        mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(mascara_limpia, contornos, -1, (255,), thickness=cv2.FILLED)
    return mascara_limpia, contornos


# ==============================================================================
# 3. PIPELINES DE SEGMENTACIÓN CON RETORNO DE SECUENCIA INTERMEDIA
# ==============================================================================
def segmentar_pvc1(img_bgr: np.ndarray) -> dict:
    blur = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    _, a, _ = cv2.split(lab)

    umbral_a, mask_otsu = cv2.threshold(
        a, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    mask_final, contornos = refinar_mascara(mask_otsu)

    return {
        "canal_a": a,
        "umbral_a": int(umbral_a),
        "mask_otsu": mask_otsu,
        "mask_final": mask_final,
        "contornos": contornos,
    }


def segmentar_pvc2(img_bgr: np.ndarray) -> dict:
    blur = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    _, a, b = cv2.split(lab)

    umbral_a, mask_a = cv2.threshold(
        a, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    umbral_b, mask_b = cv2.threshold(
        b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    mask_fusion = cv2.bitwise_or(mask_a, mask_b)
    mask_final, contornos = refinar_mascara(mask_fusion)

    # Superposición del contorno morfológico sobre la imagen RGB original
    overlay_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
    cv2.drawContours(overlay_rgb, contornos, -1, (255, 0, 255), 2)

    return {
        "umbral_a": int(umbral_a),
        "umbral_b": int(umbral_b),
        "mask_fusion": mask_fusion,
        "overlay": overlay_rgb,
        "mask_final": mask_final,
        "contornos": contornos,
    }


# ==============================================================================
# 4. EJECUCIÓN Y GENERACIÓN DE LA SECUENCIA VISUAL 2x4 SIN TRASLAPE DE TÍTULOS
# ==============================================================================
if __name__ == "__main__":
    directorio = os.path.dirname(os.path.abspath(__file__))
    ruta_pvc1 = os.path.join(directorio, "PVC1.JPG")
    ruta_pvc2 = os.path.join(directorio, "PVC2.JPG")

    img1 = cargar_imagen_segura(ruta_pvc1)
    img2 = cargar_imagen_segura(ruta_pvc2)

    if img1 is None or img2 is None:
        print(
            "[!] Error: Asegúrate de que 'PVC1.JPG' y 'PVC2.JPG' estén en la misma carpeta que este script."
        )
    else:
        res1 = segmentar_pvc1(img1)
        res2 = segmentar_pvc2(img2)

        # Guardar máscaras binarias individuales
        guardar_imagen_segura(
            os.path.join(directorio, "PVC1_binaria.png"), res1["mask_final"]
        )
        guardar_imagen_segura(
            os.path.join(directorio, "PVC2_binaria.png"), res2["mask_final"]
        )

        # Construir figura comparativa de 2 filas x 4 columnas con espaciado amplio
        fig, axes = plt.subplots(2, 4, figsize=(15, 9.0), facecolor="white")
        fig.suptitle(
            "Segmentación Óptima de Imágenes Botánicas mediante Espacios Perceptuales y Morfología Matemática\n"
            "UPMH | Maestría en Inteligencia Artificial — Sistemas de Visión Artificial (Ing. Eduardo Taurino Martínez Morales)",
            fontsize=11.5,
            fontweight="bold",
            y=0.96,
        )

        # --- FILA 1: Secuencia PVC1.JPG ---
        axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(
            "PVC1.JPG — Original (RGB)", fontsize=10, fontweight="bold", pad=8
        )

        axes[0, 1].imshow(res1["canal_a"], cmap="gray")
        axes[0, 1].set_title("1. Canal Cromático a* (CIELAB)", fontsize=10, pad=8)

        axes[0, 2].imshow(res1["mask_otsu"], cmap="gray")
        axes[0, 2].set_title(
            f"2. Otsu Inverso (T={res1['umbral_a']})", fontsize=10, pad=8
        )

        axes[0, 3].imshow(res1["mask_final"], cmap="gray")
        axes[0, 3].set_title(
            "3. Imagen Binaria Final (Óptima)", fontsize=10, fontweight="bold", pad=8
        )

        # --- FILA 2: Secuencia PVC2.JPG ---
        axes[1, 0].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(
            "PVC2.JPG — Original (RGB)", fontsize=10, fontweight="bold", pad=8
        )

        axes[1, 1].imshow(res2["mask_fusion"], cmap="gray")
        axes[1, 1].set_title(
            f"1. Fusión CIELAB a* (T={res2['umbral_a']}) ∪ b* (T={res2['umbral_b']})",
            fontsize=9.5,
            pad=8,
        )

        axes[1, 2].imshow(res2["overlay"])
        axes[1, 2].set_title(
            "2. Validación de Contorno (Morfología)", fontsize=10, pad=8
        )

        axes[1, 3].imshow(res2["mask_final"], cmap="gray")
        axes[1, 3].set_title(
            "3. Imagen Binaria Final (Óptima)", fontsize=10, fontweight="bold", pad=8
        )

        for ax in axes.flat:
            ax.axis("off")

        # Ajuste explícito de márgenes para evitar cualquier solapamiento entre filas
        plt.subplots_adjust(
            left=0.02,
            right=0.98,
            top=0.86,
            bottom=0.03,
            wspace=0.06,
            hspace=0.22,
        )

        # Guardar secuencia completa y mostrar en ventana interactiva
        ruta_secuencia = os.path.join(
            directorio, "secuencia_segmentacion_PVC1_PVC2.png"
        )
        plt.savefig(
            ruta_secuencia, dpi=200, bbox_inches="tight", pad_inches=0.15
        )
        print(f"[✓] Secuencia generada exitosamente en: {ruta_secuencia}")
        plt.show()