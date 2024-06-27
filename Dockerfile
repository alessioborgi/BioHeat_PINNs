
FROM pytorch/pytorch

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpython3-dev \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy requirements.txt and install Python dependencies
COPY requirements.txt /working_dir/requirements.txt
RUN python3 -m pip install -r /working_dir/requirements.txt

# Set environment variable for DeepXDE backend
ENV DDE_BACKEND="pytorch"

# Create and set the working directory
RUN mkdir -p /working_dir
WORKDIR /working_dir

# Copy project files into the container
COPY . /working_dir

# Default command to keep the container running
CMD ["bash"]
