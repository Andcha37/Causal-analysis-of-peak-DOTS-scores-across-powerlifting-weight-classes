"""OpenPowerlifting 원본 CSV를 보고서 분석용 남녀 CSV로 전처리한다.

기존 1~9번 전처리 스크립트의 조건을 하나로 통합한 재현용 파이프라인이다.
대용량 원본을 처리할 수 있도록 초기 필터링은 청크 단위로 수행한다.
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_INPUT = DATA_DIR / "openpowerlifting-2025-11-22-823f23d6.csv"
MIN_RECORDS = 6
DAYS_PER_YEAR = 365.2425

LIFT_COLUMNS = ["Best3DeadliftKg", "Best3BenchKg", "Best3SquatKg"]
REQUIRED_SOURCE_COLUMNS = {
    "Name",
    "Sex",
    "Equipment",
    "Tested",
    "Sanctioned",
    "Date",
    "Age",
    "BodyweightKg",
    "Place",
    "Federation",
    "ParentFederation",
    "MeetCountry",
    *LIFT_COLUMNS,
}

IPF_KEYWORDS = (
    "IPF",
    "EPF",
    "NAPF",
    "FESUPO",
    "APF",
    "ORAD",
    "JPA",
    "KPF",
    "USAPL",
    "CPU",
)

COUNTRY_TO_CONTINENT = {
    # Asia
    "Japan": "Asia",
    "South Korea": "Asia",
    "Korea": "Asia",
    "Kazakhstan": "Asia",
    "China": "Asia",
    "Taiwan": "Asia",
    "India": "Asia",
    "Philippines": "Asia",
    "Thailand": "Asia",
    "Singapore": "Asia",
    "Malaysia": "Asia",
    "Indonesia": "Asia",
    "Hong Kong": "Asia",
    "Iran": "Asia",
    "Mongolia": "Asia",
    # North America
    "USA": "North America",
    "United States": "North America",
    "Canada": "North America",
    "Mexico": "North America",
    "Puerto Rico": "North America",
    # Europe
    "Russia": "Europe",
    "Ukraine": "Europe",
    "United Kingdom": "Europe",
    "Great Britain": "Europe",
    "England": "Europe",
    "France": "Europe",
    "Germany": "Europe",
    "Italy": "Europe",
    "Sweden": "Europe",
    "Norway": "Europe",
    "Finland": "Europe",
    "Denmark": "Europe",
    "Poland": "Europe",
    "Hungary": "Europe",
    "Spain": "Europe",
    "Czechia": "Europe",
    "Ireland": "Europe",
    "Netherlands": "Europe",
    "Belgium": "Europe",
    "Austria": "Europe",
    # Oceania
    "Australia": "Oceania",
    "New Zealand": "Oceania",
    "Nauru": "Oceania",
    "Fiji": "Oceania",
    # South America
    "Brazil": "South America",
    "Argentina": "South America",
    "Colombia": "South America",
    "Chile": "South America",
    "Peru": "South America",
    "Ecuador": "South America",
    # Africa
    "South Africa": "Africa",
    "Algeria": "Africa",
    "Morocco": "Africa",
    "Egypt": "Africa",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenPowerlifting 원본을 cleaned_sss_M/F.csv로 전처리합니다."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="원본 CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="출력 폴더. 생략하면 원본 CSV와 같은 폴더",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=100_000, help="초기 처리 청크 크기"
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="중간 CSV를 _preprocess_work 폴더에 보존",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="기존 결과 파일 덮어쓰기 허용"
    )
    return parser.parse_args()


def log(stage: str, message: str) -> None:
    print(f"[{stage}] {message}", flush=True)


def validate_source(input_path: Path) -> list[str]:
    if not input_path.is_file():
        raise FileNotFoundError(f"원본 CSV를 찾을 수 없습니다: {input_path}")
    columns = list(pd.read_csv(input_path, nrows=0).columns)
    missing = sorted(REQUIRED_SOURCE_COLUMNS.difference(columns))
    if missing:
        raise ValueError(f"원본 CSV에 필요한 열이 없습니다: {missing}")
    return columns


def append_csv(df: pd.DataFrame, path: Path, first_write: bool) -> None:
    if df.empty:
        return
    df.to_csv(
        path,
        mode="w" if first_write else "a",
        header=first_write,
        index=False,
        encoding="utf-8-sig",
    )


def stage_1_raw_and_bad_names(
    input_path: Path, raw_path: Path, chunk_size: int
) -> set[str]:
    """Raw 장비만 남기고, Tested가 항상 Yes가 아닌 선수명을 수집한다."""
    bad_names: set[str] = set()
    first_write = True
    rows = 0

    for chunk in pd.read_csv(input_path, chunksize=chunk_size, low_memory=False):
        raw = chunk.loc[chunk["Equipment"].eq("Raw")].copy()
        if raw.empty:
            continue
        bad = raw.loc[raw["Tested"].ne("Yes") | raw["Tested"].isna(), "Name"]
        bad_names.update(bad.dropna().astype(str))
        append_csv(raw, raw_path, first_write)
        first_write = False
        rows += len(raw)

    if first_write:
        raise ValueError("Equipment='Raw'인 기록이 없습니다.")
    log("1/9", f"Raw 기록 {rows:,}행, Tested 부적격 선수 {len(bad_names):,}명")
    return bad_names


def stage_2_tested_sanctioned_lifts(
    raw_path: Path,
    processed_path: Path,
    bad_names: set[str],
    chunk_size: int,
) -> None:
    """Tested 선수 단위 제외, Sanctioned와 3대 유효 기록을 행 단위 필터링."""
    first_write = True
    rows = 0
    for chunk in pd.read_csv(raw_path, chunksize=chunk_size, low_memory=False):
        names = chunk["Name"].astype("string")
        chunk = chunk.loc[~names.isin(bad_names)].copy()
        chunk = chunk.loc[chunk["Sanctioned"].eq("Yes")].copy()
        for column in LIFT_COLUMNS:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        chunk = chunk.dropna(subset=LIFT_COLUMNS)
        chunk = chunk.loc[(chunk[LIFT_COLUMNS] > 0).all(axis=1)].copy()
        append_csv(chunk, processed_path, first_write)
        first_write = False if not chunk.empty else first_write
        rows += len(chunk)

    if first_write:
        raise ValueError("Tested/Sanctioned/3대 기록 필터 후 남은 데이터가 없습니다.")
    log("2/9", f"도핑·공인대회·3대 기록 필터 후 {rows:,}행")


def count_names(path: Path, chunk_size: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    for chunk in pd.read_csv(
        path, usecols=["Name"], chunksize=chunk_size, low_memory=False
    ):
        counts.update(chunk["Name"].dropna().astype(str))
    return counts


def stage_3_minimum_records(
    processed_path: Path, filtered_path: Path, chunk_size: int
) -> None:
    """선수별 전체 기록이 6개 이상인 선수만 유지한다."""
    counts = count_names(processed_path, chunk_size)
    eligible = {name for name, count in counts.items() if count >= MIN_RECORDS}
    first_write = True
    rows = 0
    for chunk in pd.read_csv(processed_path, chunksize=chunk_size, low_memory=False):
        keep = chunk["Name"].astype("string").isin(eligible)
        chunk = chunk.loc[keep].copy()
        append_csv(chunk, filtered_path, first_write)
        first_write = False if not chunk.empty else first_write
        rows += len(chunk)
    log("3/9", f"6회 이상 출전 선수 {len(eligible):,}명, {rows:,}행")


def impute_age_and_keep_adult_athletes(df: pd.DataFrame) -> pd.DataFrame:
    """나이를 날짜로 보간하고, 성인 기록이 한 번도 없는 선수를 제거한다."""
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df.sort_values(["Name", "Date"], kind="stable")

    has_any_age = df.groupby("Name", dropna=False)["Age"].transform(
        lambda values: values.notna().any()
    )
    df = df.loc[has_any_age].copy()

    df["_birth"] = pd.NaT
    known_age = df["Age"].notna() & df["Date"].notna()
    df.loc[known_age, "_birth"] = df.loc[known_age, "Date"] - pd.to_timedelta(
        df.loc[known_age, "Age"] * DAYS_PER_YEAR, unit="D"
    )
    df["_filled_birth"] = df.groupby("Name", dropna=False)["_birth"].transform(
        "first"
    )
    missing_age = df["Age"].isna() & df["_filled_birth"].notna() & df["Date"].notna()
    age_delta = df.loc[missing_age, "Date"] - df.loc[missing_age, "_filled_birth"]
    df.loc[missing_age, "Age"] = (age_delta.dt.days / DAYS_PER_YEAR).round(1)
    df = df.drop(columns=["_birth", "_filled_birth"])

    adult_names = set(df.loc[df["Age"].ge(18), "Name"].dropna().astype(str))
    df = df.loc[df["Name"].astype("string").isin(adult_names)].copy()
    log("4~5/9", f"나이 보간 및 성인 기록 보유 선수 유지 후 {len(df):,}행")
    return df


def classify_ipf(df: pd.DataFrame) -> pd.Series:
    parent = df["ParentFederation"].fillna("").astype(str).str.strip().str.upper()
    federation = df["Federation"].fillna("").astype(str).str.strip().str.upper()
    keyword_pattern = "|".join(IPF_KEYWORDS)
    is_ipf = parent.eq("IPF") | federation.str.contains(
        keyword_pattern, regex=True, na=False
    )
    return pd.Series(is_ipf.map({True: "IPF", False: "Non-IPF"}), index=df.index)


def stage_6_classification(df: pd.DataFrame) -> pd.DataFrame:
    df["IPFCategory"] = classify_ipf(df)
    country = df["MeetCountry"].fillna("").astype(str).str.strip()
    df["Continent"] = country.map(COUNTRY_TO_CONTINENT).fillna("Other/Unknown")
    log("6/9", "IPFCategory와 Continent 생성")
    return df


def stage_7_dates_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """첫 경기 후 경과일을 만들고 동일 선수·날짜 중복 기록을 병합한다."""
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    first_date = df.groupby("Name")["Date"].transform("min")
    df["Days_Since_Start"] = (df["Date"] - first_date).dt.days

    group_columns = ["Name", "Date", "Days_Since_Start"]
    aggregations: dict[str, str] = {}
    for column in df.columns:
        if column in group_columns:
            continue
        aggregations[column] = (
            "mean" if pd.api.types.is_numeric_dtype(df[column]) else "first"
        )

    df = df.groupby(group_columns, as_index=False).agg(aggregations)
    counts = df.groupby("Name")["Name"].transform("size")
    df = df.loc[counts.ge(MIN_RECORDS)].sort_values(
        ["Name", "Date"], kind="stable"
    )
    log("7/9", f"동일 날짜 병합 및 6회 이상 재필터 후 {len(df):,}행")
    return df


def stage_8_final_clean(df: pd.DataFrame) -> pd.DataFrame:
    df["BodyweightKg"] = pd.to_numeric(df["BodyweightKg"], errors="coerce")
    df = df.dropna(subset=["BodyweightKg"])
    df = df.loc[~df["Place"].isin(["DD", "DQ"])].copy()
    counts = df.groupby("Name")["Name"].transform("size")
    df = df.loc[counts.ge(MIN_RECORDS)].copy()
    log("8/9", f"체중 결측·DD·DQ 제거 및 6회 이상 재필터 후 {len(df):,}행")
    return df


def stage_9_save(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path, Path]:
    df["Sex"] = df["Sex"].astype("string").str.strip().str.upper()
    all_path = output_dir / "cleaned_sss.csv"
    male_path = output_dir / "cleaned_sss_M.csv"
    female_path = output_dir / "cleaned_sss_F.csv"

    male = df.loc[df["Sex"].eq("M")].copy()
    female = df.loc[df["Sex"].eq("F")].copy()
    df.to_csv(all_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    male.to_csv(male_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    female.to_csv(
        female_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d"
    )
    log("9/9", f"남성 {len(male):,}행 / 여성 {len(female):,}행 저장")
    return all_path, male_path, female_path


def check_outputs(output_dir: Path, overwrite: bool) -> None:
    outputs = [
        output_dir / "cleaned_sss.csv",
        output_dir / "cleaned_sss_M.csv",
        output_dir / "cleaned_sss_F.csv",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        joined = "\n  - ".join(str(path) for path in existing)
        raise FileExistsError(
            "결과 파일이 이미 있습니다. 덮어쓰려면 --overwrite를 사용하세요:\n  - "
            + joined
        )


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = (args.output_dir or input_path.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    check_outputs(output_dir, args.overwrite)
    source_columns = validate_source(input_path)
    log("준비", f"원본 {input_path} ({len(source_columns)}개 열)")

    work_dir = output_dir / "_preprocess_work"
    if work_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"중간 작업 폴더가 이미 있습니다: {work_dir}\n"
                "재실행하려면 --overwrite를 사용하세요."
            )
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    raw_path = work_dir / "raw-filtered.csv"
    processed_path = work_dir / "processed_openpowerlifting.csv"
    filtered_path = work_dir / "processed_openpowerlifting_filtered.csv"

    try:
        bad_names = stage_1_raw_and_bad_names(
            input_path, raw_path, args.chunk_size
        )
        stage_2_tested_sanctioned_lifts(
            raw_path, processed_path, bad_names, args.chunk_size
        )
        stage_3_minimum_records(processed_path, filtered_path, args.chunk_size)

        # 이 단계부터는 기존 데이터 기준 약 58MB이므로 메모리에서 처리한다.
        df = pd.read_csv(filtered_path, low_memory=False)
        df = impute_age_and_keep_adult_athletes(df)
        df = stage_6_classification(df)
        df = stage_7_dates_and_duplicates(df)
        df = stage_8_final_clean(df)
        outputs = stage_9_save(df, output_dir)

        print("\n완료:")
        for path in outputs:
            print(f"  - {path}")
    finally:
        if work_dir.exists() and not args.keep_intermediate:
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()
