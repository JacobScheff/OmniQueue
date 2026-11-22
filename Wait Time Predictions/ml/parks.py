"""The six supported parks and their isolated local workspaces."""

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
from types import SimpleNamespace
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class Park:
    slug: str
    id: str
    name: str
    timezone: str
    latitude: float
    longitude: float
    zoom: int = 16

    @property
    def tz(self):
        return ZoneInfo(self.timezone)

    def metadata(self):
        return asdict(self)


PARKS = {
    park.slug: park for park in (
        Park("disneyland", "7340550b-c14d-4def-80bb-acdb51d49a66", "Disneyland Park", "America/Los_Angeles", 33.8130, -117.9191, 17),
        Park("california-adventure", "832fcd51-ea19-4e77-85c7-75d5843b127c", "Disney California Adventure", "America/Los_Angeles", 33.8067, -117.9212, 17),
        Park("magic-kingdom", "75ea578a-adc8-4116-a54d-dccb60765ef9", "Magic Kingdom", "America/New_York", 28.4187, -81.5812),
        Park("animal-kingdom", "1c84a229-8862-4648-9c71-378ddd2c7693", "Disney's Animal Kingdom", "America/New_York", 28.3580, -81.5906),
        Park("hollywood-studios", "288747d1-8b4f-4a64-867e-ea7c9b27bad8", "Disney's Hollywood Studios", "America/New_York", 28.3575, -81.5583),
        Park("epcot", "47f90d2c-e191-4239-a466-5892ef59a88b", "EPCOT", "America/New_York", 28.3730, -81.5494),
    )
}


def get_park(value="disneyland"):
    if isinstance(value, Park):
        return value
    for park in PARKS.values():
        if value in (park.slug, park.id):
            return park
    raise ValueError(f"Unknown park {value!r}. Choose: {', '.join(PARKS)}")


def settings_for(park="disneyland", *, root=None, migrate=True, **overrides):
    from ml import config

    park = get_park(park)
    root = Path(root or config.ROOT)
    data = root / "data" / park.slug
    models = root / "checkpoints" / park.slug
    settings = {name: getattr(config, name) for name in dir(config) if name.isupper()}
    settings.update(PARK=park.slug, PARK_ID=park.id, PARK_TIMEZONE=park.tz,
                    DATASET=data / "waits.json.gz", RAW_DIR=data / "themeparks-history",
                    LOCATIONS=data / "attraction_locations.json", CHECKPOINT=models / "waits.pt")
    if migrate and park.slug == "disneyland":
        # Copy once, preserving legacy files for existing consumers and recovery.
        for old, new in ((root / "data/waits.json.gz", settings["DATASET"]),
                         (root / "data/attraction_locations.json", settings["LOCATIONS"]),
                         (root / "checkpoints/waits.pt", settings["CHECKPOINT"])):
            if new == settings["CHECKPOINT"] and (models / ".legacy-imported").exists():
                continue
            if old.exists() and not new.exists():
                new.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old, new)
            if new == settings["CHECKPOINT"] and old.exists():
                models.mkdir(parents=True, exist_ok=True)
                (models / ".legacy-imported").touch()
        old_raw = root / "data/themeparks-history"
        if old_raw.exists():
            shutil.copytree(old_raw, settings["RAW_DIR"], dirs_exist_ok=True,
                            copy_function=lambda src, dst: shutil.copy2(src, dst) if not Path(dst).exists() else dst)
    settings.update(overrides)
    return SimpleNamespace(**settings)
