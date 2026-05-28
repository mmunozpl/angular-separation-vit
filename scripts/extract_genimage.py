"""Extrae el zip multi-parte de GenImage sin crear intermedio.

Las 500 partes ``GenImage.z000..z499`` forman un único zip cuando se
concatenan en orden. En lugar de generar el zip combinado (609 GB
extra) este script las presenta a ``zipfile`` como un único stream
seekable y extrae a la carpeta destino sin duplicar nada en disco.

Modo dry-run (``--dry-run``): solo abre el zip virtual, lista cuántas
entradas hay y muestra unas pocas. Útil para validar el formato antes
de lanzar la extracción de horas.

Uso típico:
    python scripts/extract_genimage.py --dry-run
    python scripts/extract_genimage.py
"""

import argparse
import bisect
import glob
import sys
import zipfile
from pathlib import Path

from tqdm import tqdm


class ConcatFile:
    """File-like que concatena varios ficheros físicos en un stream.

    Soporta ``seek``, ``tell`` y ``read``, suficiente para que
    ``zipfile.ZipFile`` lo trate como un archivo único. Solo abre una
    parte a la vez para no exceder el límite de descriptores.
    """

    def __init__(self, paths: list[str]) -> None:
        """Indexa las partes y prepara el mapa de offsets."""
        self.paths = list(paths)
        self.sizes = [Path(p).stat().st_size for p in self.paths]
        # sumas acumuladas: cumsum[i] = offset global donde empieza
        # la parte i; cumsum[-1] = tamaño total
        self.cumsum = [0]
        for s in self.sizes:
            self.cumsum.append(self.cumsum[-1] + s)
        self.total_size = self.cumsum[-1]
        self.pos = 0
        self._fh = None
        self._cur_idx = -1

    def seekable(self) -> bool:
        """zipfile lo consulta para decidir si puede acceder al EOCD."""
        return True

    def tell(self) -> int:
        """Posición actual en el stream virtual."""
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        """Mueve la posición global; mapea a la parte adecuada al leer."""
        if whence == 1:
            offset += self.pos
        elif whence == 2:
            offset += self.total_size
        self.pos = max(0, min(int(offset), self.total_size))
        return self.pos

    def _open_part(self, idx: int) -> None:
        """Abre la parte ``idx`` si no es la actual."""
        if self._cur_idx != idx:
            if self._fh is not None:
                self._fh.close()
            self._fh = open(self.paths[idx], "rb")
            self._cur_idx = idx

    def read(self, n: int = -1) -> bytes:
        """Lee ``n`` bytes (o el resto si ``n<0``), saltando entre partes."""
        if n < 0:
            n = self.total_size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.total_size:
            idx = bisect.bisect_right(self.cumsum, self.pos) - 1
            if idx >= len(self.paths):
                break
            local = self.pos - self.cumsum[idx]
            self._open_part(idx)
            self._fh.seek(local)
            chunk = self._fh.read(min(n, self.sizes[idx] - local))
            if not chunk:
                break
            out.extend(chunk)
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def close(self) -> None:
        """Cierra el descriptor abierto, si lo hay."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def main() -> None:
    """Punto de entrada del extractor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parts-glob",
        default="/media/manpla/ST2TB/GenImage/GenImage.z*",
        help="patrón glob para localizar las partes en orden",
    )
    parser.add_argument(
        "--dest",
        default="/media/manpla/WD2TB/GenImage_extracted",
        help="carpeta destino de la extracción",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="solo abre el zip virtual y muestra estadísticas",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="extrae solo las primeras N entradas (para probar)",
    )
    args = parser.parse_args()

    parts = sorted(glob.glob(args.parts_glob))
    if not parts:
        print(f"sin partes en {args.parts_glob}")
        sys.exit(1)

    total_gb = sum(Path(p).stat().st_size for p in parts) / 2**30
    print(
        f"partes localizadas: {len(parts)} "
        f"(total {total_gb:.1f} GB)"
    )

    cf = ConcatFile(parts)
    print("abriendo zip virtual...")
    zf = zipfile.ZipFile(cf)
    members = zf.infolist()
    total_uncomp = sum(m.file_size for m in members)
    print(f"entradas en el zip: {len(members):,}")
    print(
        f"tamaño descomprimido total: "
        f"{total_uncomp / 2**30:.1f} GB"
    )
    print("primeras 5 entradas:")
    for m in members[:5]:
        print(
            f"  {m.filename}  ({m.file_size / 1024:.1f} KB, "
            f"método={m.compress_type})"
        )

    if args.dry_run:
        zf.close()
        cf.close()
        return

    Path(args.dest).mkdir(parents=True, exist_ok=True)
    to_extract = members[: args.limit] if args.limit > 0 else members
    n_ok = 0
    n_err = 0
    err_log = Path(args.dest) / "_extract_errors.log"
    with err_log.open("w") as elog:
        for member in tqdm(to_extract, desc="extrayendo"):
            try:
                zf.extract(member, args.dest)
                n_ok += 1
            except Exception as exc:
                n_err += 1
                elog.write(f"{member.filename}\t{exc}\n")
    zf.close()
    cf.close()
    print(f"\nextracción terminada: {n_ok} ok, {n_err} errores")
    if n_err > 0:
        print(f"detalle en {err_log}")
    print(f"contenido en {args.dest}")


if __name__ == "__main__":
    main()
