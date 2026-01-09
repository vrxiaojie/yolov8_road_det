FROM ubuntu:22.04


# 安装基础工具包和图形库
#RUN apt-get update && \
#    apt-get install -y \
#    git rsync jq git-lfs vim curl wget unzip lsof nload htop net-tools dnsutils openssh-server \
#    build-essential \
#    libgl1 libglib2.0-0 \
#    && \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/*


# 安装Miniconda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh   -o miniconda.sh && \
    /bin/bash miniconda.sh -b -p /opt/miniconda3 && \
    rm miniconda.sh && \
    /opt/miniconda3/bin/conda clean --all

# 设置基础环境变量
ENV PATH="/opt/miniconda3/bin:$PATH"

# 添加频道并接受服务条款
RUN conda config --add channels https://repo.anaconda.com/pkgs/main   && \
    conda config --add channels https://repo.anaconda.com/pkgs/r   && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main   && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 创建Python 3.10环境并设置为默认
RUN conda create -n py310 python=3.10 -y && \
    conda clean --all


# 更新PATH
ENV PATH="/opt/miniconda3/envs/py310/bin:${PATH}"

# 创建符号链接
RUN ln -sf /opt/miniconda3/envs/py310/bin/python /usr/local/bin/python && \
    ln -sf /opt/miniconda3/envs/py310/bin/pip /usr/local/bin/pip


# 一步到位：先用 conda 安装所有核心科学计算包 (numpy, pandas, scikit-learn, matplotlib, seaborn)
# 这确保了它们之间的二进制兼容性
RUN /bin/bash -c "source /opt/miniconda3/etc/profile.d/conda.sh && \
    conda activate py310 && \
    conda install -y numpy==1.26.4 pandas scikit-learn matplotlib seaborn tqdm==4.66.4 -c conda-forge && \
    conda install mkl==2023.1.0 polars -y \
    # 然后用 pip 安装那些 conda 中没有或您需要特定版本的包
    pip install PyYAML==6.0.1 tensorboard==2.14.0 opencv-python==4.10.0.84"

# 在py310环境中安装PyTorch (这一步现在不会破坏 numpy/scikit-learn 的兼容性)
RUN /bin/bash -c "source /opt/miniconda3/etc/profile.d/conda.sh && \
    conda activate py310 && \
    conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia && \
    conda clean --all"


# 替换为以下代码
RUN echo "source /opt/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate py310" >> ~/.bashrc && \
    /bin/bash -c "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate py310 && python --version"



# 安装VS Code服务器及扩展
RUN curl -fsSL https://code-server.dev/install.sh   | sh && \
    code-server --install-extension cnbcool.cnb-welcome && \
#    code-server --install-extension redhat.vscode-yaml && \
#    code-server --install-extension waderyan.gitblame && \
#    code-server --install-extension mhutchie.git-graph && \
#    code-server --install-extension donjayamanne.githistory && \
#    code-server --install-extension cloudstudio.live-server && \
#    code-server --install-extension tencent-cloud.coding-copilot@3.1.20 && \
    code-server --install-extension ms-python.debugpy && \
    code-server --install-extension ms-python.python


# 设置字符集支持中文
ENV LANG=C.UTF-8
ENV LANGUAGE=C.UTF-8

# 验证Python版本（可选）
RUN python --version