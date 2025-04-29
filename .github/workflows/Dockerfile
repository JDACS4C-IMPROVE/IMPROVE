# Use the correct base image (change based on your architecture)
FROM condaforge/mambaforge:latest

# Set the working directory
WORKDIR /app

# Install necessary system packages
RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

# Update Conda and use mamba for dependency resolution
RUN conda install -n base -c conda-forge mamba \
    && conda update -n base -c defaults conda \
    && conda config --add channels conda-forge \
    && conda config --set channel_priority flexible

# Fix missing dependencies (_libgcc_mutex issue)
RUN conda install -n base -c conda-forge libgcc-ng libstdcxx-ng --yes

# Clean Conda cache
RUN conda clean --all --yes

# Clone the GraphDRP repository
RUN git clone https://github.com/JDACS4C-IMPROVE/GraphDRP.git

# Change to the GraphDRP directory
WORKDIR /app/GraphDRP

# Use mamba to create the Conda environment (faster and more reliable)
RUN mamba env create -f conda_env.yml --platform=linux-64

# Ensure Conda is sourced correctly
SHELL ["/bin/bash", "-c"]

# Activate the Conda environment on startup
RUN echo "conda activate graphdrp" >> ~/.bashrc

# Source the setup_improve.sh script (on container start)
RUN echo "source setup_improve.sh" >> ~/.bashrc

# Download the dataset (skip download if it exists)
RUN wget --cut-dirs=8 -P ./ -nH -np -m -N https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/

# Set the entrypoint to bash
ENTRYPOINT ["/bin/bash"]