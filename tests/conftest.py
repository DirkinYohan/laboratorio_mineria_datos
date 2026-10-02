"""Configuración compartida de la suite de pruebas.

Añade la raíz del proyecto a ``sys.path`` para que los tests puedan importar
el paquete ``src`` aunque se ejecuten desde otro directorio.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
