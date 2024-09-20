

from pyannote.audio import Pipeline
from transformers import pipeline



def segmentation(audio):

    asr = pipeline(
        "automatic-speech-recognition",
        model="facebook/wav2vec2-large-960h-lv60-self",
        feature_extractor="facebook/wav2vec2-large-960h-lv60-self",

    )
    speaker_segmentation = Pipeline.from_pretrained("pyannote/speaker-segmentation")

    speaker_output = speaker_segmentation(audio)
    text_output = asr(audio, return_timestamps="word",
                      chunk_length_s=10, stride_length_s=(4, 2))
    # https://huggingface.co/blog/asr-chunking

    full_text = text_output['text'].lower()
    chunks = text_output['chunks']

    diarized_output = ""
    i = 0

    resulted = {'speaker': [],
                'text': [],
                'start_second': [],
                'end_second': []}
    for turn, _, speaker in speaker_output.itertracks(yield_label=True):
        diarized = ""
        while i < len(chunks) and chunks[i]['timestamp'][1] <= turn.end:
            diarized += chunks[i]['text'] + ' '
            # diarized += chunks[i]['text'].lower() + ' '
            i += 1

        if diarized != "":
            # diarized_output += "{}: ''{}'' from {:.3f}-{:.3f}\n".format(speaker, diarized, turn.start, turn.end)
            resulted['speaker'].append(speaker)
            resulted['text'].append(diarized)
            resulted['start_second'].append(turn.start)
            resulted['end_second'].append(turn.end)

    return resulted

"""
import os
import sys

if sys.platform == 'win32':
    sep = ';'
else:
    sep = ':'
"""
# os.environ['PATH'] += sep + r'"C:\Users\Edward\Downloads\ffmpeg-master-latest-win64-gpl\bin"'

# v = './air_traffic_control_audio.wav'
# v = 'Ray Dalio on US Dominance China Economy Inflation Future of Bridgewater.mp4'
# result = segmentation(audio=v)

"""
ar = f"{1}"
ac = "1"
format_for_conversion = "f32le"
ffmpeg_command = [
    "ffmpeg",
    "-i",
    "pipe:0",
    "-ac",
    ac,
    "-ar",
    ar,
    "-f",
    format_for_conversion,
    "-hide_banner",
    "-loglevel",
    "quiet",
    "pipe:1",
]
"""
# import subprocess
"""
try:
    with subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as ffmpeg_process:
        print('hello world')
except Exception as e:
    raise e
"""

# subprocess.Popen('ffmpeg')


# KeysView(environ({'ALLUSERSPROFILE': 'C:\\ProgramData', 'APPDATA': 'C:\\Users\\Edward\\AppData\\Roaming', 'CHOCOLATEYINSTALL': 'C:\\ProgramData\\chocolatey', 'CHOCOLATEYLASTPATHUPDATE': '132946182117623448', 'COMMONPROGRAMFILES': 'C:\\Program Files\\Common Files', 'COMMONPROGRAMFILES(X86)': 'C:\\Program Files (x86)\\Common Files', 'COMMONPROGRAMW6432': 'C:\\Program Files\\Common Files', 'COMPUTERNAME': 'LAPTOP-TQ4KH194', 'COMSPEC': 'C:\\Windows\\system32\\cmd.exe', 'DRIVERDATA': 'C:\\Windows\\System32\\Drivers\\DriverData', 'HOMEDRIVE': 'C:', 'HOMEPATH': '\\Users\\Edward', 'IDEA_INITIAL_DIRECTORY': 'C:\\Program Files\\JetBrains\\PyCharm Community Edition 2022.2\\bin', 'IPYTHONENABLE': 'True', 'LOCALAPPDATA': 'C:\\Users\\Edward\\AppData\\Local', 'LOGONSERVER': '\\\\LAPTOP-TQ4KH194', 'NUMBER_OF_PROCESSORS': '12', 'ONEDRIVE': 'C:\\Users\\Edward\\OneDrive', 'ONEDRIVECONSUMER': 'C:\\Users\\Edward\\OneDrive', 'OS': 'Windows_NT', 'PATH': 'C:\\TET\\env\\video_summary\\Scripts;C:\\Program Files\\Python310\\Scripts\\;C:\\Program Files\\Python310\\;C:\\Windows\\system32;C:\\Windows;C:\\Windows\\System32\\Wbem;C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\;C:\\Windows\\System32\\OpenSSH\\;C:\\ProgramData\\chocolatey\\bin;C:\\Program Files\\Git\\cmd;C:\\Program Files\\dotnet\\;C:\\Users\\Edward\\AppData\\Local\\Microsoft\\WindowsApps;C:\\Users\\Edward\\Documents\\geckodriver', 'PATHEXT': '.COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC;.PY;.PYW', 'PROCESSOR_ARCHITECTURE': 'AMD64', 'PROCESSOR_IDENTIFIER': 'AMD64 Family 23 Model 104 Stepping 1, AuthenticAMD', 'PROCESSOR_LEVEL': '23', 'PROCESSOR_REVISION': '6801', 'PROGRAMDATA': 'C:\\ProgramData', 'PROGRAMFILES': 'C:\\Program Files', 'PROGRAMFILES(X86)': 'C:\\Program Files (x86)', 'PROGRAMW6432': 'C:\\Program Files', 'PROMPT': '(video_summary) $P$G', 'PSMODULEPATH': 'C:\\Program Files\\WindowsPowerShell\\Modules;C:\\Windows\\system32\\WindowsPowerShell\\v1.0\\Modules', 'PUBLIC': 'C:\\Users\\Public', 'PYCHARM_HOSTED': '1', 'PYDEVD_LOAD_VALUES_ASYNC': 'True', 'PYTHONPATH': 'C:/Program Files/JetBrains/PyCharm Community Edition 2022.2/plugins/python-ce/helpers/third_party/thriftpy;C:/Program Files/JetBrains/PyCharm Community Edition 2022.2/plugins/python-ce/helpers/pydev;C:\\TET\\video_summary;C:\\TET\\video_summary', 'PYTHONUNBUFFERED': '1', 'SESSIONNAME': 'Console', 'SYSTEMDRIVE': 'C:', 'SYSTEMROOT': 'C:\\Windows', 'TEMP': 'C:\\Users\\Edward\\AppData\\Local\\Temp', 'TMP': 'C:\\Users\\Edward\\AppData\\Local\\Temp', 'USERDOMAIN': 'LAPTOP-TQ4KH194', 'USERDOMAIN_ROAMINGPROFILE': 'LAPTOP-TQ4KH194', 'USERNAME': 'Edward', 'USERPROFILE': 'C:\\Users\\Edward', 'VIRTUAL_ENV': 'C:\\TET\\env\\video_summary', 'WINDIR': 'C:\\Windows', '_OLD_VIRTUAL_PATH': 'C:\\Program Files\\Python310\\Scripts\\;C:\\Program Files\\Python310\\;C:\\Windows\\system32;C:\\Windows;C:\\Windows\\System32\\Wbem;C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\;C:\\Windows\\System32\\OpenSSH\\;C:\\ProgramData\\chocolatey\\bin;C:\\Program Files\\Git\\cmd;C:\\Program Files\\dotnet\\;C:\\Users\\Edward\\AppData\\Local\\Microsoft\\WindowsApps;C:\\Users\\Edward\\Documents\\geckodriver', '_OLD_VIRTUAL_PROMPT': '$P$G', 'KMP_INIT_AT_FORK': 'FALSE', 'KMP_DUPLICATE_LIB_OK': 'True'}))
#          environ({'ALLUSERSPROFILE': 'C:\\ProgramData', 'APPDATA': 'C:\\Users\\Edward\\AppData\\Roaming', 'CHOCOLATEYINSTALL': 'C:\\ProgramData\\chocolatey', 'CHOCOLATEYLASTPATHUPDATE': '132946182117623448', 'COMMONPROGRAMFILES': 'C:\\Program Files\\Common Files', 'COMMONPROGRAMFILES(X86)': 'C:\\Program Files (x86)\\Common Files', 'COMMONPROGRAMW6432': 'C:\\Program Files\\Common Files', 'COMPUTERNAME': 'LAPTOP-TQ4KH194', 'COMSPEC': 'C:\\Windows\\system32\\cmd.exe', 'DRIVERDATA': 'C:\\Windows\\System32\\Drivers\\DriverData', 'HOMEDRIVE': 'C:', 'HOMEPATH': '\\Users\\Edward', 'IDEA_INITIAL_DIRECTORY': 'C:\\Program Files\\JetBrains\\PyCharm Community Edition 2022.2\\bin', 'IPYTHONENABLE': 'True', 'LOCALAPPDATA': 'C:\\Users\\Edward\\AppData\\Local', 'LOGONSERVER': '\\\\LAPTOP-TQ4KH194', 'NUMBER_OF_PROCESSORS': '12', 'ONEDRIVE': 'C:\\Users\\Edward\\OneDrive', 'ONEDRIVECONSUMER': 'C:\\Users\\Edward\\OneDrive', 'OS': 'Windows_NT', 'PATH': 'C:\\TET\\env\\video_summary\\Scripts;C:\\Program Files\\Python310\\Scripts\\;C:\\Program Files\\Python310\\;C:\\Windows\\system32;C:\\Windows;C:\\Windows\\System32\\Wbem;C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\;C:\\Windows\\System32\\OpenSSH\\;C:\\ProgramData\\chocolatey\\bin;C:\\Program Files\\Git\\cmd;C:\\Program Files\\dotnet\\;C:\\Users\\Edward\\Downloads\\ffmpeg-master-latest-win64-gpl\\bin;C:\\Users\\Edward\\AppData\\Local\\Microsoft\\WindowsApps;C:\\Users\\Edward\\Documents\\geckodriver;', 'PATHEXT': '.COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC;.PY;.PYW', 'PROCESSOR_ARCHITECTURE': 'AMD64', 'PROCESSOR_IDENTIFIER': 'AMD64 Family 23 Model 104 Stepping 1, AuthenticAMD', 'PROCESSOR_LEVEL': '23', 'PROCESSOR_REVISION': '6801', 'PROGRAMDATA': 'C:\\ProgramData', 'PROGRAMFILES': 'C:\\Program Files', 'PROGRAMFILES(X86)': 'C:\\Program Files (x86)', 'PROGRAMW6432': 'C:\\Program Files', 'PROMPT': '(video_summary) $P$G', 'PSMODULEPATH': 'C:\\Program Files\\WindowsPowerShell\\Modules;C:\\Windows\\system32\\WindowsPowerShell\\v1.0\\Modules', 'PUBLIC': 'C:\\Users\\Public', 'PYCHARM_HOSTED': '1', 'PYDEVD_LOAD_VALUES_ASYNC': 'True', 'PYTHONPATH': 'C:/Program Files/JetBrains/PyCharm Community Edition 2022.2/plugins/python-ce/helpers/third_party/thriftpy;C:/Program Files/JetBrains/PyCharm Community Edition 2022.2/plugins/python-ce/helpers/pydev;', 'PYTHONUNBUFFERED': '1', 'SESSIONNAME': 'Console', 'SYSTEMDRIVE': 'C:', 'SYSTEMROOT': 'C:\\Windows', 'TEMP': 'C:\\Users\\Edward\\AppData\\Local\\Temp', 'TMP': 'C:\\Users\\Edward\\AppData\\Local\\Temp', 'USERDOMAIN': 'LAPTOP-TQ4KH194', 'USERDOMAIN_ROAMINGPROFILE': 'LAPTOP-TQ4KH194', 'USERNAME': 'Edward', 'USERPROFILE': 'C:\\Users\\Edward', 'VIRTUAL_ENV': 'C:\\TET\\env\\video_summary', 'WINDIR': 'C:\\Windows', '_OLD_VIRTUAL_PATH': 'C:\\Program Files\\Python310\\Scripts\\;C:\\Program Files\\Python310\\;C:\\Windows\\system32;C:\\Windows;C:\\Windows\\System32\\Wbem;C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\;C:\\Windows\\System32\\OpenSSH\\;C:\\ProgramData\\chocolatey\\bin;C:\\Program Files\\Git\\cmd;C:\\Program Files\\dotnet\\;C:\\Users\\Edward\\Downloads\\ffmpeg-master-latest-win64-gpl\\bin;C:\\Users\\Edward\\AppData\\Local\\Microsoft\\WindowsApps;C:\\Users\\Edward\\Documents\\geckodriver;', '_OLD_VIRTUAL_PROMPT': '$P$G'})