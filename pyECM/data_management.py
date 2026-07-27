from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException

import time
import os
import zipfile
import tempfile
import pathlib


def obtain_nearest_structure(
    xyz_FILENAME,
    input_path,
    output_path=None,
    download_dir=None,
    silent=True,
    remove_files=False,
):

    # Check if the input and output paths are valid
    input_path = str(pathlib.Path(input_path).resolve())
    output_path = str(pathlib.Path(output_path).resolve())

    if output_path is None:
        output_path = input_path

    if download_dir is None:
        with tempfile.TemporaryDirectory() as download_dir:
            download_CSM_zip(input_path, xyz_FILENAME, download_dir, silent)
            split_output(download_dir + "/results.zip", output_path + "/", xyz_FILENAME)
    else:
        if os.path.exists(download_dir + "/results.zip"):
            raise FileExistsError(
                "The file 'results.zip' already exists in the download directory. "
                "Please remove it before running again."
            )
        download_CSM_zip(input_path, xyz_FILENAME, download_dir, silent)
        split_output(download_dir + "/results.zip", output_path + "/", xyz_FILENAME)

    if remove_files:
        os.remove(input_path + "/" + xyz_FILENAME + ".xyz")


def download_CSM_zip(input_path, xyz_FILENAME, download_dir, silent=True):
    # Initialize the browser in the following order: Chrome, Firefox, Edge, Safari
    driver = None

    try:
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_experimental_option(
            "prefs",
            {
                # Configure the download folder
                "download.default_directory": download_dir,
                # Do not ask before downloading
                "download.prompt_for_download": False,
                "directory_upgrade": True,
            },
        )
        if silent:
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(options=chrome_options)
    except WebDriverException:
        try:
            firefox_options = webdriver.FirefoxOptions()
            firefox_profile = webdriver.FirefoxProfile()
            firefox_profile.set_preference(
                "browser.download.folderList", 2
            )  # Personalized folder
            firefox_profile.set_preference("browser.download.dir", download_dir)
            firefox_profile.set_preference(
                "browser.helperApps.neverAsk.saveToDisk", "application/zip"
            )
            if silent:
                firefox_options.add_argument("--headless")
            driver = webdriver.Firefox(
                firefox_profile=firefox_profile, options=firefox_options
            )
        except WebDriverException:
            try:
                edge_options = webdriver.EdgeOptions()
                edge_options.add_experimental_option(
                    "prefs",
                    {
                        # Configure the download folder
                        "download.default_directory": download_dir,
                        # Do not ask before downloading
                        "download.prompt_for_download": False,
                        "directory_upgrade": True,
                    },
                )
                if silent:
                    edge_options.add_argument("--headless")
                driver = webdriver.Edge(options=edge_options)
            except WebDriverException:
                try:
                    driver = webdriver.Safari()
                except WebDriverException:
                    print(
                        "There is no supported browser found. "
                        "Make sure you have Chrome, Firefox, Edge or Safari installed."
                    )
                    exit(1)

    # Navigate to the website
    driver.get("https://csm.ouproj.org.il/molecule")

    # Wait until the file input is clickable
    wait = WebDriverWait(driver, 5)
    upload_button = wait.until(EC.element_to_be_clickable((By.ID, "file")))
    upload_button.send_keys(input_path + "/" + xyz_FILENAME + ".xyz")

    # Click on "CCM"
    ccm_button = wait.until(
        EC.element_to_be_clickable((By.XPATH, '//button[text()=" CCM:"]'))
    )
    ccm_button.click()

    # Wait until the select element is visible
    wait = WebDriverWait(driver, 10)
    select_element = wait.until(EC.presence_of_element_located((By.TAG_NAME, "select")))

    # Create a Select object and select the option with value "20"
    select = Select(select_element)
    select.select_by_value("20")

    # Wait until the "Calculate" button is clickable and click on it
    calculate_button = wait.until(
        EC.element_to_be_clickable((By.CLASS_NAME, "btn-success"))
    )
    calculate_button.click()

    # Wait unil the "Download class result" button is clickable and click on it
    download_button = wait.until(
        EC.element_to_be_clickable((By.CLASS_NAME, "btn-primary"))
    )
    download_button.click()

    # Wait until the file is downloaded
    while True:
        # Check if the file has been downloaded
        if os.path.exists(download_dir + "/results.zip"):
            break
        # Wait for 1 second before checking again
        time.sleep(1)


def split_output(zip_file, output_dir=None, mol_name="molecule"):

    # Extract the desired file in the specified directory
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extract("resulting_mols.xyz", path=output_dir)

    # Open the original file in read mode
    with open(output_dir + "resulting_mols.xyz", "r") as file:
        lines = file.readlines()

    # Calculate the mid point
    mid_point = len(lines) // 2

    # Split the lines into two parts
    part1 = lines[:mid_point]
    part2 = lines[mid_point:]

    # Write the first part in a new file
    with open(output_dir + mol_name + "_chiral.xyz", "w") as file:
        file.writelines(part1)

    # Write the second part in a new file
    with open(output_dir + mol_name + "_achiral.xyz", "w") as file:
        file.writelines(part2)

    # Eliminate the original file
    os.remove(output_dir + "resulting_mols.xyz")

    # Eliminate the zip file
    os.remove(zip_file)
