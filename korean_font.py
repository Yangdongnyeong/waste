"""한글 폰트 설정 유틸리티 (Windows / Linux / macOS)"""

import os
import platform

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


def set_korean_font():
    """Windows / Linux(Colab 포함) / macOS 환경에 맞춰 한글 폰트를 설정합니다."""
    system = platform.system()

    if system == "Windows":
        font_path = "c:/Windows/Fonts/malgun.ttf"
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rc("font", family=font_name)
        print(f"한글 폰트 설정: {font_name} ({font_path})")

    elif system == "Linux":
        # 1) 이미 설치된 나눔 폰트 탐색
        nanum_paths = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/nanum/NanumGothic.ttf",
            os.path.expanduser("~/.local/share/fonts/NanumGothic.ttf"),
        ]
        font_path = next((p for p in nanum_paths if os.path.isfile(p)), None)

        # 2) 없으면 apt-get으로 설치 시도 (Colab / Ubuntu)
        if font_path is None:
            print("나눔 폰트 설치 중 (apt-get)...")
            import subprocess

            subprocess.run(
                ["apt-get", "install", "-y", "-qq", "fonts-nanum"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            font_path = next((p for p in nanum_paths if os.path.isfile(p)), None)

        # 3) apt-get 불가 시 직접 다운로드 (홈 디렉터리에 저장)
        if font_path is None:
            print("나눔 폰트 다운로드 중...")
            import urllib.request

            font_dir = os.path.expanduser("~/.local/share/fonts")
            os.makedirs(font_dir, exist_ok=True)
            font_path = os.path.join(font_dir, "NanumGothic.ttf")
            url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
            urllib.request.urlretrieve(url, font_path)

        if font_path and os.path.isfile(font_path):
            fm.fontManager.addfont(font_path)
            font_name = fm.FontProperties(fname=font_path).get_name()
            plt.rc("font", family=font_name)
            print(f"한글 폰트 설정: {font_name} ({font_path})")
        else:
            print("경고: 한글 폰트를 설정할 수 없습니다.")

    elif system == "Darwin":  # macOS
        plt.rc("font", family="AppleGothic")
        print("한글 폰트 설정: AppleGothic")
