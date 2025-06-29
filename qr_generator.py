# qr_generator.py
"""Genera los PNG con los códigos QR para cada reto de foto.

Uso:
  python qr_generator.py --base https://tusitio.com

Generará 3 archivos:
  qr_reto_1.png, qr_reto_2.png, qr_reto_3.png

Opciones:
  --base   Dominio/base URL donde corre tu app (sin la barra final)
  --eq     (opcional) nombre del equipo para inyectarlo como parámetro
           Ej.: --eq=espartanos generará .../foto_reto/1?eq=espartanos
"""

import qrcode
import argparse
from urllib.parse import urlencode


def make_qr(url: str, outfile: str):
    img = qrcode.make(url)
    img.save(outfile)
    print(f"🆗  {outfile} → {url}")


def main():
    parser = argparse.ArgumentParser(description="Genera códigos QR para los retos de foto")
    parser.add_argument("--base", required=True, help="URL base, ej. https://networking-sxxt.onrender.com")
    parser.add_argument("--eq",   help="Nombre de equipo para agregar como parámetro")
    args = parser.parse_args()

    for n in (1, 2, 3):
        path = f"/foto_reto/{n}"
        params = {"eq": args.eq} if args.eq else {}
        url = args.base.rstrip("/") + path + ("?" + urlencode(params) if params else "")
        make_qr(url, f"qr_reto_{n}.png")

    print("\n✅  QR creados. Imprime los PNG y pégalos en las estaciones.")


if __name__ == "__main__":
    main()
