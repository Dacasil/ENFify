"""Module to generate PDF files and plots"""

import os

import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF


# Class to generate a PDF file
class PDF(FPDF):
    """_summary_

    Args:
        FPDF (_type_): _description_
    """

    def header(self):
        self.set_font("Arial", "B", 18)
        self.cell(
            0, 10, "Audio tampering detection via the electrical network frequency", 0, 1, "C"
        )

    def chapter_title(self, title):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, title, 0, 1, "C")
        self.ln(10)

    def chapter_body(self, body):
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def chapter_body_with_bold(self, body):
        parts = body.split("\033[1m")  # Split at bold indicator
        for part in parts:
            if "\033[0m" in part:
                bold_text, normal_text = part.split("\033[0m")
                self.set_font("Arial", "B", 12)
                self.multi_cell(0, 10, bold_text)
                self.set_font("Arial", "", 12)
                self.multi_cell(0, 10, normal_text)
            else:
                self.set_font("Arial", "", 12)
                self.multi_cell(0, 10, part)
        self.ln()

    def add_image(self, image_path, x=None, y=None, w=0, h=0):
        self.image(image_path, x, y, w, h)
        self.ln(10)

    def add_images_side_by_side(self, image1_path, image2_path, image_width, image_height):
        y = 200
        self.image(image1_path, x=10, y=y, w=image_width)
        # self.image(image2_path, x=self.w / 2 + 10, w=image_width)
        self.image(image2_path, x=100, y=y, w=image_width)
        self.ln(image_width + 10)

    def add_text_and_image(self, text, image_path, image_width):
        # Text on the left
        self.set_xy(10, self.get_y())
        self.multi_cell(self.w / 2 - 15, 10, text)

        # Image on the right
        self.set_xy(self.w / 2 + 10, self.get_y() - (len(text.split("\n")) * 10))
        self.image(image_path, w=image_width)
        self.ln(image_width + 10)


def create_cut_phase_plot(x_new, phases_new, x_old, region_of_interest, path):
    """_summary_

    Args:
        x_new (_type_): _description_
        phases_new (_type_): _description_
        x_old (_type_): _description_
        region_of_interest (_type_): _description_
        path (_type_): _description_
    """

    plt.plot(x_new, np.degrees(phases_new))
    plt.xlabel("Time in seconds")
    plt.ylabel("Phase (degrees)")
    plt.title("Phasediagramm")
    plt.axvline(x=x_old[int(np.min(region_of_interest))] - 1, color="red", linestyle="--")
    plt.axvline(x=x_old[int(np.max(region_of_interest))] + 1, color="red", linestyle="--")
    plt.grid(True)
    plt.savefig(path)
    plt.close()


def create_phase_plot(x_new, phases_new, path):
    """_summary_

    Args:
        x_new (_type_): _description_
        phases_new (_type_): _description_
        path (_type_): _description_
    """
    plt.plot(x_new, np.degrees(phases_new))
    plt.xlabel("Time in seconds")
    plt.ylabel("Phase in degrees")
    plt.title("Phasediagramm")
    plt.grid(True)
    plt.savefig(path)
    plt.close()


def cut_to_alpha_pdf(im_path1, im_paths, outpath):
    """_summary_

    Args:
        im_path1 (_type_): _description_
        im_path2 (_type_): _description_
    """
    pdf = PDF()
    pdf.add_page()

    # Diagramm in PDF einfügen
    pdf.chapter_title("Hilbert phase estimation")
    pdf.add_image(im_path1, w=pdf.w - 20)
    pdf.chapter_body(
        "The Plot shows the phase diagram of the input signal. The phase is estimated via instantaneous Hilbert phase estimation."
    )
    pdf.chapter_body_with_bold("\033[1mThe Audio shows discontinuities in its phase\033[0m")
    pdf.chapter_body_with_bold("The discontinuities and areas of interest are displayed in the following")

    for i in range(len(im_paths)):
        pdf.add_page()
        pdf.chapter_title("Hilbert phase discontinuities")
        pdf.add_image(im_paths[i], w=pdf.w - 20)
        pdf.chapter_body(
            "The Plot shows the phase diagram of the input signal in a found zone of interest."
        )

    # PDF speichern
    pdf.output(outpath)
    print(f"Results pdf saved at {os.path.normpath(outpath)}")


def to_alpha_pdf(im_path, outpath):
    """_summary_

    Args:
        im_path1 (_type_): _description_
        im_path2 (_type_): _description_
    """
    pdf = PDF()
    pdf.add_page()

    # Diagramm in PDF einfügen
    pdf.chapter_title("Hilbert phase estimation")
    pdf.add_image(im_path, w=pdf.w - 20)
    pdf.chapter_body(
        "The Plot shows the phase diagram of the input signal. The phase is estimated via instantaneous Hilbert phase estimation."
    )
    pdf.chapter_body_with_bold("\033[1mThe Audio shows no discontinuities in its phase\033[0m")

    # PDF speichern
    pdf.output(outpath)
    print(f"Results pdf saved at {os.path.normpath(outpath)}")


def plot_func(x_hilbert, x_hilbert_new, hilbert_phases, hilbert_phases_new, hil_interest_region):

    # Create the phase plots
    # TODO: Paths in config or as terminal arguments
    root_path = os.path.join(os.path.dirname(__file__), "../reports")
    hilbert_phase_path = f"{root_path}/figures/hilbert_phase_im.png"
    pdf_outpath = f"{root_path}/enfify_alpha.pdf"

    if not np.any(hil_interest_region):
        create_phase_plot(x_hilbert, hilbert_phases, hilbert_phase_path)
        to_alpha_pdf(hilbert_phase_path, pdf_outpath)

    else:
        create_phase_plot(x_hilbert, hilbert_phases, hilbert_phase_path)

        cut_hil_phase_paths = []
        for i in range(len(hil_interest_region)):
            cut_hil_phase_paths.append(f"{root_path}/figures/{i}cut_hilbert_phase_im.png")

            create_cut_phase_plot(
                x_hilbert_new[i],
                hilbert_phases_new[i],
                x_hilbert,
                hil_interest_region[i],
                cut_hil_phase_paths[i],
            )
        
        cut_to_alpha_pdf(hilbert_phase_path, cut_hil_phase_paths, pdf_outpath)
