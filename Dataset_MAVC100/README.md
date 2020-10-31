<h2> MAVC100 </h2>
Please download the original video files of the MAVC100 from here, the corresponding frame level audio labels and frame level audio-visual labels. Please note that the difference between audio labels and audio-visual labels is: 
<p> Audio labels contain 4 classes events: Silence, Speech, Singing, and Others. Speech and Singing contain all speech and singing voice in audio streams, without distinguishing whether the speech or singing voice comes from the anchor (the target speaker) or background;
<p> Audio-visual labels also contain 4 classes events: Silence, Speech, Singing, and Others. But Speech and Singing in Audio-visual labels just contain the speech and singing voice from anchor (target speaker), and are different from the Speech and Singing in audio labels which may from background sounds.

<h2> Label explanation</h2>
1. Regardless of whether it is an audio label or an audio-visual label, we have only marked three types of labels: speech, singing and silence. The remaining time in the clip will be considered as others. Therefore, when using labels, you need to add the label of others after calculating the free time segment yourself.
2. The labeling is performed by marking the start time position and the end time position when each event appears, and different classes are represented by different numbers:
<p> Speech: represented by 1;
<p> Singing: represented by 2;
<p> Silence: represented by 3;
<p> Starting time position: represented by s;
<p> Ending time position: denoted by e;
For example:
1.	Starting time position of Singing: 2s;
2.	Ending time position of Speech: 1e;
3.	Starting time position of Silence: 3s.

<h2> Description of each file or folder</h2>
1. The ¡°Source_urls_of_live_streams.txt¡± contains the original link of all the data used, you can choose to use the original link to download the video data, or download it directly in the folder (Backup_of_source_video_streams) on this page;
2. ¡°Backup_of_source_video_streams¡± contains the original video data that we have backed up and organized. You can choose to download it directly;
3. ¡°Frame_level_audio_labels.rar¡± is the frame-level labels in audio streams;
4. ¡°Frame_level_audio-visual_labels.rar¡± includes the audio-visual frame-level labels.

Please note that the original video data of the MAVC100 comes from the Internet, and the label data are all made by ourselves. The MAVC100 can only be used for research purposes and not for other purposes. If the original video data is infringing, please contact us and we will delete this backup in time.

The annotations of the MAVC100 can be used for research free of charge, but we reserve the corresponding legal rights.

Mail: Yuanbo.Hou@UGent.be
