from pathlib import Path
import unittest

import geopandas as gpd
import pandas as pd


ROOT = Path(__file__).resolve().parent
FIGURE2_DATA = ROOT / "data" / "figure2"
SHAPEFILES = (
    FIGURE2_DATA / "HK_border" / "hk_merged_border.shp",
    FIGURE2_DATA / "HK_maintenance" / "Maintenance_project.shp",
    FIGURE2_DATA / "Toronto_border" / "citygcs_regional_mun_wgs84.shp",
    FIGURE2_DATA / "LA_border" / "City_Boundary.shp",
)


class Figure2InputTests(unittest.TestCase):
    def test_shapefiles_are_complete_and_readable(self):
        for shapefile in SHAPEFILES:
            with self.subTest(shapefile=shapefile.name):
                for suffix in (".shp", ".shx", ".dbf", ".prj"):
                    sidecar = shapefile.with_suffix(suffix)
                    self.assertTrue(sidecar.is_file())
                    self.assertGreater(sidecar.stat().st_size, 0)
                frame = gpd.read_file(shapefile)
                self.assertFalse(frame.empty)
                self.assertIsNotNone(frame.crs)

    def test_measured_workbook_contains_wwtp_flow(self):
        workbook = FIGURE2_DATA / "17WWTP_inflow_data1.xlsx"
        with pd.ExcelFile(workbook) as excel:
            self.assertIn("WWTP flow", excel.sheet_names)
            frame = pd.read_excel(excel, sheet_name="WWTP flow")
        self.assertFalse(frame.empty)


if __name__ == "__main__":
    unittest.main()
