from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from scipy.spatial import cKDTree
from shapely.geometry import Point, MultiPoint, LineString, MultiLineString, GeometryCollection, shape
from shapely.ops import unary_union

app = FastAPI(title="BOSS map API")

app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:5500",
    #                 "http://127.0.0.1:5500"],  # или ["*"] для теста
    allow_origins=["*"],  # или ["*"] для теста
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_FILE = "paths.config.json"
SEARCH_DIR = Path(".")

def find_single_file(prefix: str, suffix: str = "") -> str:
    matches = [
        p.name for p in SEARCH_DIR.iterdir()
        if p.is_file()
        and p.name.startswith(prefix)
        and (p.name.endswith(suffix) if suffix else True)
    ]

    if len(matches) == 0:
        raise FileNotFoundError(f"No file found starting with '{prefix}'")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple files found for '{prefix}': {matches}")

    return matches[0]


# ---------- MAIN LOGIC ----------

if Path(CONFIG_FILE).exists():
    with open(CONFIG_FILE, encoding="utf-8") as f:
        cfg = json.load(f)

    CSV_PATH = cfg["CSV_PATH"]
    ABONENT_CSV_PATH = cfg["ABONENT_CSV_PATH"]
    ROADS_RIVERS_GEOJSON_PATH = cfg["ROADS_RIVERS_GEOJSON_PATH"]

else:
    CSV_PATH = find_single_file("copper_distrib", ".csv")
    ABONENT_CSV_PATH = find_single_file("addresses", ".csv")
    ROADS_RIVERS_GEOJSON_PATH = find_single_file("roads_rivers", ".geojson")


print("Using files:")
print("CSV_PATH:", CSV_PATH)
print("ABONENT_CSV_PATH:", ABONENT_CSV_PATH)
print("ROADS_RIVERS_GEOJSON_PATH:", ROADS_RIVERS_GEOJSON_PATH)

# В CSV нет заголовков, поэтому задаём свои
COLS = [
    "id",         # 0
    "code",       # 1 (например 50602/36)
    "rk",         # 2 (РК)
    "status",     # 3 (Рабочий)
    "address",    # 5
    "lon",        # 6
    "lat",        # 7
    "ports",  # 8
    "len",   # 9 (например "218.500 м")
    "place",       # 10 (например "Внутреннее Этаж2")
    "id_rsh",
    "code_rsh",
    "status_rsh",
    "address_rsh",
    "lon_rsh",
    "lat_rsh",
    "force_external" # заставить подключатся к ИЖС
]

MAX_DISTANCE = 200

# words to remove entirely
WORDS_TO_REMOVE = ["проспект", "улица", "блок", "переулок", "трасса", "шоссе", "микрорайон"]

TYPES_EXTERNAL = ["IZHS"]
SKIPPED_TYPES_INTERNAL = ["IZHS"]

# regex for full-word removal
REMOVE_WORDS_PATTERN = r'(^|[\s,])(?:{})($|[\s,])'.format(
    "|".join(map(re.escape, WORDS_TO_REMOVE))
)
# remove trailing "а" from every word
TRAILING_A_PATTERN = r"(^|[\s,])([0-9А-Яа-яәіңғүұқөһӘІҢҒҮҰҚӨ]+?)(а|-й)(?=[\s,]|$)"

ENG_TO_CYR = str.maketrans({
    "a": "а",
    "b": "б",
    "c": "к",   # or "ц" depending on system — chosen standard
    "d": "д",
    "e": "е",
    "f": "ф",
    "g": "г",
    "h": "х",
    "i": "и",
    "j": "дж",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "п",
    "q": "к",
    "r": "р",
    "s": "с",
    "t": "т",
    "u": "у",
    "v": "в",
    "w": "в",
    "x": "кс",
    "y": "й",
    "z": "з"
})

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"[\u200b\u200c\u200d\ufeff\u2060\u180e]", "", s)

    if "," in s:
        s = s.rsplit(",", 1)[1]

    s = s.lower()

    # remove everything in parentheses ()
    s = re.sub(r"\([^)]*\)", " ", s)

    # remove words like "улица", "проспект"
    s = re.sub(REMOVE_WORDS_PATTERN, " ", s)

    # remove trailing "а" from ALL words
    s = re.sub(TRAILING_A_PATTERN, r"\1\2", s)

    # normalize spaces
    s = re.sub(r"\s+", "", s).strip()

    return s

def safe_json_response(payload):
    """
    Ensures JSON-serializable output.
    If serialization fails, prints the offending path and object.
    """
    try:
        json.dumps(payload, ensure_ascii=False)
        return JSONResponse(payload)
    except TypeError as e:
        print("❌ JSON serialization error:", e)

        def find_bad(obj, path="root"):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    find_bad(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    find_bad(v, f"{path}[{i}]")
            else:
                try:
                    json.dumps(obj, ensure_ascii=False)
                except Exception:
                    print("🚨 Non-serializable value at:", path)
                    print("   Type:", type(obj))
                    print("   Value:", repr(obj))

        find_bad(payload)
        raise

def normalize_house(s: str) -> str:
    if not isinstance(s, str):
        return ""

    s = re.sub(r"[\u200b\u200c\u200d\ufeff\u2060\u180e]", "", s)

    s = s.lower()

    # replace English letters with Cyrillic
    s = s.translate(ENG_TO_CYR)

    # remove words like "блок"
    s = re.sub(REMOVE_WORDS_PATTERN, " ", s)

    # normalize spaces
    s = re.sub(r"\s+", "", s).strip()

    return s

def split_address(addr: str):
    if not isinstance(addr, str) or not addr.strip():
        return "", ""

    # split by LAST comma
    if "," in addr:
        street, house = addr.rsplit(",", 1)
    else:
        street, house = addr, ""

    return normalize_text(street), normalize_house(house)

def levenshtein(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + (ca != cb)
            )
            prev, dp[j] = dp[j], cur
    return dp[-1]

def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    dist = levenshtein(a, b)
    return 1 - dist / max(len(a), len(b))

def street_score(street_rk: str, street_abon: str) -> float:
    ta = street_rk.split()
    tb = street_abon.split()

    if not ta or not tb:
        return 0.0

    score = 0.0
    for wa in ta:
        best = max(similarity(wa, wb) for wb in tb)
        score += best

    return score / len(ta)

roads = []
road_shapes = []
rivers = []
river_shapes = []
with open(ROADS_RIVERS_GEOJSON_PATH, encoding="utf-8") as f:
    data = json.load(f)

for feat in data.get("features", []):
    props = feat.get("properties", {}) or {}
    geom = feat.get("geometry", {}) or {}

    row = {
        "type": "Feature",
        "geometry": {
            "type": geom.get("type"),
            "coordinates": geom.get("coordinates")
        }
    }
    
    if props.get("waterway","") != "":
        rivers.append(row)
        river_shapes.append(shape(row["geometry"]))
    elif props.get("name", "") != "" and geom.get("type").endswith("LineString"):
    # elif props.get("highway","") in ["primary", "secondary", "service"]:
        roads.append(row)
        road_shapes.append(shape(row["geometry"]))

river_geom = unary_union(river_shapes)
road_geom = unary_union(road_shapes)

def intersects_twice(line: LineString, geom) -> bool:
    inter = line.intersection(geom)

    if inter.is_empty:
        return False

    # Multiple distinct points
    if isinstance(inter, MultiPoint):
        return len(inter.geoms) >= 2

    # Touching at exactly one point
    if isinstance(inter, Point):
        return False

    # Line overlaps road (bad → treat as crossing twice)
    if inter.geom_type in ("LineString", "MultiLineString"):
        return True

    # GeometryCollection → count points inside it
    if isinstance(inter, GeometryCollection):
        points = [g for g in inter.geoms if isinstance(g, Point)]
        return len(points) >= 2

    return False

def parseGeometry(wkt: str):
    if not wkt or not isinstance(wkt, str):
        return None

    wkt = wkt.strip()
    wkt_upper = wkt.upper()

    def parse_coords(text):
        coords = []
        for pair in text.split(","):
            pair = pair.strip()
            if not pair:
                continue
            parts = pair.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid coordinate pair: '{pair}'")
            lon, lat = map(float, parts)
            coords.append([lon, lat])
        return coords

    # ---------- POINT ----------
    if wkt_upper.startswith("POINT"):
        inner = wkt[wkt.find("(") + 1 : wkt.rfind(")")]
        lon, lat = map(float, inner.split())
        return {
            "type": "Point",
            "coordinates": [lon, lat]
        }

    # ---------- LINESTRING ----------
    if wkt_upper.startswith("LINESTRING"):
        inner = wkt[wkt.find("(") + 1 : wkt.rfind(")")]
        return {
            "type": "LineString",
            "coordinates": parse_coords(inner)
        }

    # ---------- POLYGON ----------
    if wkt_upper.startswith("POLYGON"):
        inner = wkt[wkt.find("((") + 2 : wkt.rfind("))")]
        rings = []

        for ring in inner.split("),("):
            coords = parse_coords(ring)
            if coords:
                rings.append(coords)

        return {
            "type": "Polygon",
            "coordinates": rings
        }

    # ---------- MULTIPOLYGON ----------
    if wkt_upper.startswith("MULTIPOLYGON"):
        inner = wkt[wkt.find("(((") + 3 : wkt.rfind(")))")]
        polygons = []

        for polygon_text in inner.split(")),(("):
            rings = []
            for ring in polygon_text.split("),("):
                coords = parse_coords(ring)
                if coords:
                    rings.append(coords)
            if rings:
                polygons.append(rings)

        return {
            "type": "MultiPolygon",
            "coordinates": polygons
        }

    raise ValueError(f"Unsupported WKT geometry: {wkt}")

def group_rk_by_coords(df_devices: pd.DataFrame):
    df_devices = df_devices.copy()
    df_devices["rk_group_id"] = None

    rk_groups = []
    group_id = 1
    name_counters = {}

    # group by identical lon + lat
    for (lon, lat), group in df_devices.groupby(["lon", "lat"], dropna=False):

        if len(group) <= 1:
            continue

        # base name from code_rsh (take first row arbitrarily)
        base_name = group.iloc[0]["code_rsh"]

        # handle duplicate names → "|| 2", "|| 3", ...
        count = name_counters.get(base_name, 0) + 1
        name_counters[base_name] = count

        rk_group_name = base_name if count == 1 else f"{base_name} || {count}"

        # create rk group row
        rk_groups.append({
            "id": str(group_id),
            "lon": lon,
            "lat": lat,
            "rk_group_name": rk_group_name,
            "address": group.iloc[0]["address"],
        })

        # assign rk_group_id back to devices
        df_devices.loc[group.index, "rk_group_id"] = str(group_id)

        group_id += 1

    df_rk_groups = pd.DataFrame(rk_groups)

    return df_devices, df_rk_groups

def extract_cabinet_info(df_devices):
    # one row per cabinet
    cabinets = (
        df_devices
        .drop_duplicates(subset=["id_rsh"])
        .reset_index(drop=True)
    )

    # --- df_cabinets ---
    df_cabinets = pd.DataFrame({
        "id": cabinets["id_rsh"],
        "boss_id": "",
        # remove leading "РШ " from code
        "cabinet_number": cabinets["code_rsh"].str.replace(r"^РШ\s*", "", regex=True),
        "cabinet_name": cabinets["code_rsh"],
        "address": cabinets["address"],
        "lon": cabinets["lon_rsh"],
        "lat": cabinets["lat_rsh"],
        # generate cabinet_region_id (stable, unique)
        "cabinet_region_id": cabinets["id_rsh"],
    })

    # --- df_cabinet_regions ---
    df_cabinet_regions = pd.DataFrame({
        "id": df_cabinets["cabinet_region_id"],
        "name": df_cabinets["cabinet_name"],
    })

    return df_cabinets, df_cabinet_regions

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def address_matching_assignment(df_devices, df_addresses):
    df_devices = df_devices.copy()
    df_addresses = df_addresses.copy()

    # normalization
    df_devices[["_street_lc", "_house_lc"]] = (
        df_devices["address"]
        .fillna("")
        .apply(lambda x: pd.Series(split_address(x)))
    )
    df_addresses["_street_lc"] = df_addresses["street"].fillna("").apply(normalize_text)
    df_addresses["_house_lc"] = df_addresses["house"].fillna("").apply(normalize_house)

    internal_rk_mask = (
        df_devices["place"].fillna("").str.startswith("Внутреннее") &
        ~df_devices["force_external"] &
        ~df_devices["status"].fillna("").isin(["offline","pending"])
    )

    # --- assignment ---
    for addr_idx, r in df_addresses.iterrows():
        street_lc = r["_street_lc"]
        house_lc = r["_house_lc"]
        coords = parseGeometry(r["location"])["coordinates"]

        if (not street_lc or len(r["kya_rk"]) != 0 or r["building_type"] in SKIPPED_TYPES_INTERNAL) and not r["force_one_to_one"]:
            continue

        if r["name"] == "Кабанбай батыр проспект, 19 блок E" or r["name"] == "Кабанбай батыр проспект, 19 блок С":
            df_addresses.at[addr_idx, "kya_rk"].append("16372991")
            df_addresses.at[addr_idx, "kya_rk"].append("16395049")
            df_addresses.at[addr_idx, "code_rsh"].append("РШ 24019")
            df_devices.at[df_devices.index[df_devices["id"] == "16372991"][0], "abonents"].append(r["id"])
            df_devices.at[df_devices.index[df_devices["id"] == "16395049"][0], "abonents"].append(r["id"])
            continue
        
        matches = df_devices[
            internal_rk_mask &
            (df_devices["abonents"].str.len() == 0) &
            (abs(df_devices["lon"]-coords[0]) <= MAX_DISTANCE/38093.0) &
            (abs(df_devices["lat"]-coords[1]) <= MAX_DISTANCE/110574.3)
        ]

        # collect matching df indices directly into result
        result = []

        # split address name by all " / "
        name_parts = [
            p.strip()
            for p in re.split(r"\s+/\s+", str(r["name"]))
            if "," in p
        ]

        for part in name_parts:
            try:
                s, h = part.split(",", 1)
            except ValueError:
                continue

            street_alt = normalize_text(s)
            house_alt = normalize_house(h)

            tmp = matches[matches["_house_lc"] == house_alt].copy()
            if tmp.empty:
                continue

            tmp["_score"] = tmp["_street_lc"].apply(
                lambda n: street_score(n, street_alt)
            )

            tmp = tmp[tmp["_score"] >= 0.6]

            for m_idx, match in tmp.sort_values(by="_score", ascending=False).iterrows():
                d = haversine(
                    float(match["lat"]),
                    float(match["lon"]),
                    coords[1],
                    coords[0]
                )
                if d <= MAX_DISTANCE:
                    result.append(m_idx)

        # single-address logic
        matches2 = matches[matches["_house_lc"] == house_lc].copy()
        matches2["_score"] = matches2["_street_lc"].apply(
            lambda n: street_score(n, street_lc)
        )
        matches2 = matches2[matches2["_score"] >= 0.6]

        for m_idx, match in matches2.sort_values(by="_score", ascending=False).iterrows():
            d = haversine(
                float(match["lat"]),
                float(match["lon"]),
                coords[1],
                coords[0]
            )
            if d <= MAX_DISTANCE:
                result.append(m_idx)

        # deduplicate while preserving order
        result = list(dict.fromkeys(result))

        if not result:
            continue

        # geometry correction if needed
        poly = shape(parseGeometry(r["area"]))
        for df_idx in result:
            df_addresses.at[addr_idx, "kya_rk"].append(df_devices.at[df_idx, "id"])
            if df_devices.at[df_idx, "code_rsh"] not in df_addresses.at[addr_idx, "code_rsh"]:
                df_addresses.at[addr_idx, "code_rsh"].append(df_devices.at[df_idx, "code_rsh"])
            if not poly.contains(Point(df_devices.at[df_idx, "lon"], df_devices.at[df_idx, "lat"])):
                df_devices.at[df_idx, "lon"] = coords[0]
                df_devices.at[df_idx, "lat"] = coords[1]
            # append abonent
            df_devices.at[df_idx, "abonents"].append(r["id"])

    # ---- cleanup ----
    df_devices.drop(columns=["_street_lc", "_house_lc"], inplace=True)
    df_addresses.drop(columns=["_street_lc", "_house_lc"], inplace=True)

    return df_devices, df_addresses

def fast_kya_rk_assignment(df_devices, df_addresses):
    df_devices = df_devices.copy(deep=True)
    df_addresses = df_addresses.copy(deep=True)

    df_devices["abonents"] = df_devices["abonents"].apply(list)
    df_addresses["kya_rk"] = df_addresses["kya_rk"].apply(list)
    df_addresses["code_rsh"] = df_addresses["code_rsh"].apply(list)

    # ---- parse coords ----
    df_devices["_coord"] = df_devices.apply(lambda r: (r["lon"], r["lat"]), axis=1)
    df_addresses["_coord"] = df_addresses["location"].apply(parseGeometry)

    # skip internal RK/KYA
    external_rk_mask = (
        (~df_devices["place"].fillna("").str.startswith("Внутреннее") |
        df_devices["force_external"]) &
        ~df_devices["status"].fillna("").isin(["offline","pending"])
    )

    # ---- build KD-tree for RK/KYA ----
    rk_df = df_devices[external_rk_mask].copy()
    rk_coords = np.array(rk_df["_coord"].tolist())
    rk_tree = cKDTree(rk_coords)
    rk_pos_to_dfidx = rk_df.index.to_numpy()

    # ---- select candidates ----
    candidates = []

    for addr_idx, a in df_addresses.iterrows():
        # skip addresses already assigned
        if len(a["kya_rk"]) > 0 or (a["building_type"] not in TYPES_EXTERNAL) or a["force_one_to_one"]:
            continue

        lon, lat = a["_coord"]["coordinates"]
        addr_pt = Point(lon, lat)

        # query nearest RK within radius
        idxs = rk_tree.query_ball_point([lon, lat], r=MAX_DISTANCE / 111_000)
        
        for pos in idxs:
            df_idx = rk_pos_to_dfidx[pos]
            r_lon, r_lat = rk_coords[pos]
            d = haversine(lat, lon, r_lat, r_lon)
            if d > MAX_DISTANCE:
                continue  # too far
            rk_pt = Point(r_lon, r_lat)
            link_line = LineString([addr_pt, rk_pt])
            if link_line.intersects(river_geom):
                continue  # ❌ crosses river
            if intersects_twice(link_line, road_geom):
                continue  # ❌ crosses road twice
            candidates.append((addr_idx, df_idx, d))

    cand_df = pd.DataFrame(candidates, columns=["addr_idx", "df_idx", "distance"])
    if cand_df.empty:
        # no candidates, return original frames
        df_devices.drop(columns="_coord", inplace=True)
        df_addresses.drop(columns="_coord", inplace=True)
        return df_devices, df_addresses

    # ---- sort abonents by nearest RK ----
    addr_order = (
        cand_df.groupby("addr_idx")["distance"]
        .min()
        .sort_values()
        .index
        .tolist()
    )

    # ---- greedy assignment ----
    for addr_idx in addr_order:
        rows = cand_df[cand_df["addr_idx"] == addr_idx].sort_values("distance")
        for _, row in rows.iterrows():
            df_idx = row["df_idx"]
            if int(df_devices.at[df_idx, "ports"]) > len(df_devices.at[df_idx, "abonents"]):
                df_devices.at[df_idx, "abonents"].append(df_addresses.at[addr_idx, "id"])
                df_addresses.at[addr_idx, "kya_rk"].append(df_devices.at[df_idx, "id"])
                df_addresses.at[addr_idx, "code_rsh"].append(df_devices.at[df_idx, "code_rsh"])
                break  # assigned, move to next address

    # ---- cleanup ----
    df_devices.drop(columns="_coord", inplace=True)
    df_addresses.drop(columns="_coord", inplace=True)

    return df_devices, df_addresses

@app.get("/objects.geojson")
def objects_geojson(limit: int = 50000):
    try:
        with open("changes.json", encoding="utf-8") as f:
            changes = json.load(f)
    except FileNotFoundError:
        changes = None
        print("changes.json not found, no changes made")
    
    df_addresses = pd.read_csv(ABONENT_CSV_PATH, encoding="utf-8", dtype="string")
    df_addresses = df_addresses.dropna(subset=["location", "area"])
    df_addresses["kya_rk"] = [[] for _ in range(len(df_addresses))]
    df_addresses["code_rsh"] = [[] for _ in range(len(df_addresses))]
    if "force_one_to_one" in df_addresses.columns:
        df_addresses["force_one_to_one"] = df_addresses["force_one_to_one"].astype(str).str.strip().str.lower().eq("true")
    else:
        df_addresses["force_one_to_one"] = False
    if changes and changes.get("newAddresses"):
        new_addr = df_addresses["id"].map(changes["newAddresses"])
        split_addr = new_addr.str.split(",", n=1, expand=True)
        df_addresses["street"] = split_addr[0].str.strip().fillna(df_addresses["street"])
        df_addresses["house"] = split_addr[1].str.strip().fillna(df_addresses["house"])
    df_other = df_addresses[
        df_addresses["street"].isna() | df_addresses["house"].isna() |
        (df_addresses["street"].str.strip() == "") | (df_addresses["house"].str.strip() == "")
    ]
    df_addresses = df_addresses.dropna(subset=["house", "street"])

    df_devices = pd.read_csv(CSV_PATH, encoding="utf-8", header=None, names=COLS, dtype="string")
    df_devices["status"] = "active"
    if changes and changes.get("lon"):
        df_devices["lon"] = df_devices["id"].map(changes["lon"]).fillna(df_devices["lon"])
    if changes and changes.get("lat"):
        df_devices["lat"] = df_devices["id"].map(changes["lat"]).fillna(df_devices["lat"])
    if changes and changes.get("lon_rsh"):
        df_devices["lon_rsh"] = df_devices["id"].map(changes["lon_rsh"]).fillna(df_devices["lon_rsh"])
    if changes and changes.get("lat_rsh"):
        df_devices["lat_rsh"] = df_devices["id"].map(changes["lat_rsh"]).fillna(df_devices["lat_rsh"])
    if changes and changes.get("place"):
        df_devices["place"] = df_devices["id"].map(changes["place"]).fillna(df_devices["place"])
    df_devices["lat"] = pd.to_numeric(df_devices["lat"], errors="coerce")
    df_devices["lon"] = pd.to_numeric(df_devices["lon"], errors="coerce")
    df_devices["lat_rsh"] = pd.to_numeric(df_devices["lat_rsh"], errors="coerce")
    df_devices["lon_rsh"] = pd.to_numeric(df_devices["lon_rsh"], errors="coerce")
    df_devices["force_external"] = df_devices["force_external"].fillna("false")
    df_devices["force_external"] = df_devices["force_external"].astype(str).str.strip().str.lower().eq("true")
    df_devices = df_devices.dropna(subset=["lat", "lon"])
    df_devices["abonents"] = [[] for _ in range(len(df_devices))]
    
    if changes and changes.get("editedAddresses"):
        df_devices["address"] = df_devices["id"].map(changes["editedAddresses"]).fillna(df_devices["address"])
    if changes and changes.get("editedForceExternal"):
        df_devices["force_external"] = df_devices["id"].map(changes["editedForceExternal"]).astype("boolean").fillna(df_devices["force_external"])
    if changes and changes.get("editedForce1To1"):
        df_addresses["force_one_to_one"] = df_addresses["id"].map(changes["editedForce1To1"]).astype("boolean").fillna(df_addresses["force_one_to_one"])

    df_devices, df_addresses = address_matching_assignment(df_devices, df_addresses)

    if changes and changes.get("editedRsh"):
        df_addresses["code_rsh"] = df_addresses["id"].map(changes["editedRsh"]).fillna(df_addresses["code_rsh"])
    if changes and changes.get("editedStatuses"):
        df_devices["status"] = df_devices["id"].map(changes["editedStatuses"]).fillna(df_devices["status"])

    df_devices, df_addresses = fast_kya_rk_assignment(df_devices, df_addresses)
    print(f"Есть {df_devices['abonents'].apply(len).eq(0).sum()} РК без подключения")

    df_cabinets, df_cabinet_regions = extract_cabinet_info(df_devices)
    df_devices, df_rk_groups = group_rk_by_coords(df_devices)

    rk = []
    for _, r in df_devices.iterrows():
        rk.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
            "properties": {
                "id": str(r["id"]),
                # "boss_id": "",
                "rk_number": str(r["code"]),
                "rk_group_id": r["rk_group_id"],
                "address_id": r["abonents"],
                "rk_name": str(r["rk"]),
                "place": str(r["place"]),
                "rk_external": str(r["place"]).startswith("Внешнее") or r["force_external"],
                "status": str(r["status"]),
                "address": str(r["address"]),
                "ports_number": str(r["ports"]),
                "distance_to_cabinet": str(r["len"]),
                "cabinet_id": str(r["id_rsh"]),
                "code_rsh": str(r["code_rsh"]),  # delete
                "status_rsh": str(r["status_rsh"]),  # delete
                "address_rsh": str(r["address_rsh"]),  # delete
                "coords_rsh": [r["lon_rsh"], r["lat_rsh"]],  # delete
            }
        })

    address = []
    for _, r in df_addresses.iterrows():
        try:
            geom = parseGeometry(r["area"])
            point_geom = parseGeometry(r["location"])
        except Exception:
            continue
        
        address.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "id": str(r.get("id", "")),
                "two_gis_id": str(r.get("id", "")),
                "yandex_id": "",
                "name": str(r.get("name", "")),
                "location": point_geom,
                "address_izhs": str(r["building_type"]) == "IZHS" and not r["force_one_to_one"],
                "region": str(r.get("region", "")),
                "locality": str(r.get("locality", "")),
                "district": str(r.get("district", "")),
                "street": str(r.get("street", "")),
                "house": str(r.get("house", "")),
                "network_type": str(r.get("network_type", "")),
                "building_type": str(r.get("building_type", "")),
                "purpose": str(r.get("purpose", "")),
                "residential_complex": str(r.get("residential_complex", "")),
                "residential_complex_completed": str(r.get("res_completed", "")),
                "houses_count": str(r.get("houses_count", "")),
                "apartments_count": str(r.get("apartments_count", "")),
                "floors_count": str(r.get("floors_count", "")),
                "porch_count": str(r.get("porch_count", "")),
                "cluster_id": str(r.get("cluster_id", "")),
                "kya_rk": r["kya_rk"],  # delete
                "code_rsh": r["code_rsh"],  # delete
            }
        })

    other = []
    for _, r in df_other.iterrows():
        try:
            geom = parseGeometry(r["area"])
            point_geom = parseGeometry(r["location"])
        except Exception:
            continue
        
        other.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "id": str(r.get("id", "")),
                "name": str(r.get("name", "")),
                "location": point_geom,
                "building_type": str(r.get("building_type", ""))
            }
        })

    cabinet = []
    for _, r in df_cabinets.iterrows():
        cabinet.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
            "properties": {
                "id": str(r["id"]),
                "boss_id": str(r["boss_id"]),
                "cabinet_number": str(r["cabinet_number"]),
                "cabinet_name": str(r["cabinet_name"]),
                "address": str(r["address"]),
                "cabinet_region_id": str(r["cabinet_region_id"]),
            }
        })
    
    cabinet_region = []
    for _, r in df_cabinet_regions.iterrows():
        cabinet_region.append({
            "type": "Feature",
            "geometry": {"type": "MultiPolygon", "coordinates": None},
            "properties": {
                "id": str(r["id"]),
                "name": r["name"],
            }
        })
    
    rk_group = []
    for _, r in df_rk_groups.iterrows():
        rk_group.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
            "properties": {
                "id": str(r["id"]),
                "rk_group_name": str(r["rk_group_name"]),
                "address": str(r["address"]),
            }
        })

    return safe_json_response({"type": "FeatureCollection", "rk": rk, "address": address, "cabinet": cabinet, "rk_group": rk_group, "cabinet_region": cabinet_region, "other": other, "changes": changes})

@app.post("/skip-izhs")
def skip_izhs(ids: dict):
    """
    Add skipped IZH IDs to skipped_izhs.json (append, not overwrite)
    Expected format: {"ids": ["id1", "id2", ...]}
    """
    try:
        new_ids = ids.get("ids", [])
        
        if not isinstance(new_ids, list):
            return JSONResponse(
                status_code=400,
                content={"error": "ids must be a list"}
            )
        
        filepath = Path("../frontend/skipped_izhs.json")
        
        # Read existing IDs
        existing_ids = []
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    existing_ids = json.load(f) or []
                    if not isinstance(existing_ids, list):
                        existing_ids = []
            except:
                existing_ids = []
        
        # Merge with new IDs (avoid duplicates)
        existing_set = set(existing_ids)
        for id_val in new_ids:
            existing_set.add(id_val)
        
        merged_ids = sorted(list(existing_set))
        
        # Save merged list
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(merged_ids, f, ensure_ascii=False, indent=2)
        
        return JSONResponse(
            status_code=200,
            content={"success": True, "message": f"Added {len(new_ids)} ID(s), total {len(merged_ids)} skipped IZH IDs", "count": len(merged_ids)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/skipped_izhs.json")
def get_skipped_izhs():
    """
    Get current skipped IZH IDs from skipped_izhs.json
    """
    try:
        filepath = Path("../frontend/skipped_izhs.json")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(
            status_code=200,
            content=data,
            media_type="application/json"
        )
    except FileNotFoundError:
        return JSONResponse(
            status_code=200,
            content=[],
            media_type="application/json"
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

