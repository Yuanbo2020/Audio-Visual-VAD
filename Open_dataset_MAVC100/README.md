<h2> MAVC100 </h2>
Please download the original video files of the MAVC100 from here, the corresponding frame level audio labels and frame level audio-visual labels. Please note that the difference between audio labels and audio-visual labels is: 

<p><li>Audio labels contain 4 classes events: Silence, Speech, Singing, and Others. Speech and Singing contain all speech and singing voice in audio streams, without distinguishing whether the speech or singing voice comes from the anchor (the target speaker) or background;</li></p>

<p><li>Audio-visual labels also contain 4 classes events: Silence, Speech, Singing, and Others. But Speech and Singing in Audio-visual labels just contain the speech and singing voice from anchor (target speaker), and are different from the Speech and Singing in audio labels which may from background sounds.</li></p>

<h2> Label explanation</h2>
<p>1. Regardless of whether it is an audio label or an audio-visual label, we have only marked three types of labels: speech, singing and silence. The remaining time in the clip will be considered as others. Therefore, when using labels, you need to add the label of others after calculating the free time segment yourself.</p>

<p>2. The labeling is performed by marking the start time position and the end time position when each event appears, and different classes are represented by different numbers:</p>
<p><li>Speech: represented by 1;</li></p>
<p><li>Singing: represented by 2;</li></p>
<p><li>Silence: represented by 3;</li></p>
<p><li>Starting time position: represented by s;</li></p>
<p><li>Ending time position: denoted by e.</li></p>

For example:

<p><li>Starting time position of Singing: 2s;</li></p>

<p><li>Ending time position of Speech: 1e;</li></p>

<p><li>Starting time position of Silence: 3s.</li></p>

<h2> Description of each file or folder</h2>
<p><li>The 'Source_urls_of_live_streams.txt' contains the original link of all the data used, you can choose to use the original link to download the video data, or download it directly in the folder (Backup_of_source_video_streams) on this page;</li></p>

<p><li>'Backup_of_source_video_streams' contains the original video data that we have backed up and organized. You can choose to download it directly;</li></p>

<p><li>'Frame_level_audio_labels.rar' is the frame-level labels in audio streams;</li></p>

<p><li>'Frame_level_audio-visual_labels.rar' includes the audio-visual frame-level labels.</li></p>

<p>Please note that the original video data of the MAVC100 comes from the Internet, and the label data are all made by ourselves. The MAVC100 can only be used for research purposes and not for other purposes. If the original video data is infringing, please contact us and we will delete this backup in time.</p>

<p>The annotations of the MAVC100 can be used for research free of charge, but we reserve the corresponding legal rights.</p>

# Citation
Please feel free to use this data set and consider citing our paper as

```bibtex
@inproceedings{icassp2021_hou,
  author    = {Yuanbo Hou and
               Yi Deng and
               Bilei Zhu and
               Zejun Ma and
               Dick Botteldooren},
  title     = {Rule-embedded network for audio-visual voice activity detection in
               live musical video streams},
  booktitle = {{IEEE} International Conference on Acoustics, Speech and Signal Processing,
               {ICASSP} 2021},
  publisher = {{IEEE}},
  year      = {2021},
}
```

