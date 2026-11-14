#!/usr/bin/env python3
"""
Export de la base GénéFoncier (lecture seule) vers un GeoPackage pour
construire le dataset d'entraînement.

Couches produites :
  - parcelles : parcelles_typology historiques (date < 1900, non supprimées)
  - feuilles  : communes_feuilles + section + commune (résolue spatialement si
                le lien section -> commune manque) + stats de couverture
  - sections  : communes_sections
  - communes  : communes_geom

Usage (python système, celui de QGIS, qui a GDAL) :
    python3 tools/export_bdd.py --env /chemin/GeneFoncier/.env \
        --out data/gis_export/genefoncier.gpkg
"""
import argparse, re
from pathlib import Path
from osgeo import gdal, ogr

gdal.UseExceptions(); ogr.UseExceptions()

RURAL = "('labour','prairie','bois','jardin','marais','pature','verger','vigne','friche')"

SQL = {
    "parcelles": """
        select par_typ_id::text as id, type, type2, parcelles_no, parcelles_bis,
               date_start, date_end, par_main_typ::text as par_main_id, geom
        from parcelles_typology
        where deleted_at is null and geom is not null
          and (date_start < '1900-01-01' or date_start is null)  -- non datées = saisies anciennes (Roclincourt, Farbus...)
    """,
    "feuilles": f"""
        with f as (
          select f.com_feuille_id, f.feuille_no, f.feuille_name, f.date_start, f.date_end,
                 s.section_letter, s.section_name, g.code_insee, g.commune_old_name, f.geom
          from communes_feuilles f
          left join communes_sections s on s.com_sections_id = f.com_sections_id
          left join communes_geom g on g.com_geometry_id = s.com_geometry_id
          where f.geom is not null
        ), c as (  -- commune résolue spatialement quand le lien manque
          select distinct on (f.com_feuille_id) f.com_feuille_id, g.code_insee, g.commune_old_name
          from f join communes_geom g on st_intersects(g.geom, st_pointonsurface(f.geom))
          order by f.com_feuille_id, abs(extract(year from g.date_start) - extract(year from f.date_start)) nulls last
        ), p as (
          select f.com_feuille_id, count(t.*) n_parcelles,
                 count(t.*) filter (where t.type in {RURAL}) n_rural,
                 coalesce(sum(st_area(t.geom)), 0) / nullif(st_area(f.geom), 0) as couverture
          from f left join parcelles_typology t
            on t.deleted_at is null and t.date_start < '1900-01-01'
           and extract(year from t.date_start) = extract(year from f.date_start)
           and st_intersects(f.geom, st_pointonsurface(t.geom))
          group by f.com_feuille_id, f.geom
        )
        select f.com_feuille_id::text as id, coalesce(f.code_insee, c.code_insee) as code_insee,
               coalesce(f.commune_old_name, c.commune_old_name) as commune,
               f.section_letter as section, f.section_name, f.feuille_no, f.feuille_name,
               f.date_start, f.date_end, p.n_parcelles, p.n_rural,
               round(p.couverture::numeric, 3)::float as couverture, f.geom
        from f left join c using (com_feuille_id) left join p using (com_feuille_id)
    """,
    "sections": """
        select com_sections_id::text as id, section_letter, section_name, date_start, date_end,
               com_geometry_id::text as commune_id, geom
        from communes_sections where geom is not null
    """,
    "communes": """
        select com_geometry_id::text as id, code_insee, commune_old_name, date_start, date_end, geom
        from communes_geom where geom is not null
    """,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help=".env de GeneFoncier (DB_HOST, DB_NAME, ...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", help="surcharge DB_HOST (ex: 127.0.0.1 via tunnel SSH)")
    ap.add_argument("--port", help="surcharge DB_PORT (ex: 5434 via tunnel SSH)")
    a = ap.parse_args()
    env = dict(re.findall(r"^(\w+)=(.*)$", Path(a.env).read_text(), re.M))
    g = lambda k, d=None: env.get(k, d).strip().strip('"')
    conn = (f"PG:host={a.host or g('DB_HOST')} port={a.port or g('DB_PORT', '5432')} dbname={g('DB_NAME')} "
            f"user={g('DB_USER')} password={g('DB_PASSWORD')}")
    src = ogr.Open(conn)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists(): out.unlink()
    dst = ogr.GetDriverByName("GPKG").CreateDataSource(str(out))
    for name, sql in SQL.items():
        lyr = src.ExecuteSQL(sql)
        dst.CopyLayer(lyr, name)
        print(f"  {name:10} {lyr.GetFeatureCount():7d} entités")
        src.ReleaseResultSet(lyr)
    dst = None
    print(f"✅ {out}")


if __name__ == "__main__":
    main()
