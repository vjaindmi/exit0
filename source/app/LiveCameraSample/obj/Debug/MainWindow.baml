   M S B A M L   `   `   `  ����K  GLiveCameraSample, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null3clr-namespace:LiveCameraSampleLiveCameraSample  5   	   !    LiveCameraSample.MainWindow   5   	   R NWindowsBase, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35\ XPresentationFramework, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35W SPresentationCore, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35D 9http://schemas.microsoft.com/winfx/2006/xaml/presentation    5   	   R NSystem.Xaml, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     5   	   @d2http://schemas.microsoft.com/expression/blend/2008     5   	   Jmc;http://schemas.openxmlformats.org/markup-compatibility/2006     5   	   0localclr-namespace:LiveCameraSample     5   	     � Title$  Auspex��5   	   $	��600��6   $	��750��6%   .��5	      �  � ColumnDefinitions 5
   
   �� 5      $	��11* �6   �� 5      $
��300* �6   �� 5      $
��406* �6    � RowDefinitions 5   
   �� 5      $��40 �6   �� 5      $��* �6   �� 5      $��40 �6   .+�5   
   �� -   5        $��	ButtonBar��5   ;    �� Orientation$ 
Horizontal=�6   $��0��6.   $��3��6L   .+�5      �� -   5         ��Name$ 
CameraList��5   .   $y�	Collapsed=�6   $	��100��6B   $��5,5,0,0q�6N   6_   �� 5      -   5        $��Start Camera��5      ���#CE353D
�������$��35��6T   $	��160��6`   
�������$	��5,0q�6�   $��5,5,0,0q�6�   6�   �� 5      -   5        $��Stop Camera��5      ���#CE353D
�������$��35��6T   $	��160��6`   
�������$	��5,0q�6�   $��5,5,0,0q�6�   6�   �� 5      $��Center=�6   $y�	Collapsed=�63   $��15,5,0,5q�6J   $��Mode:��6\   �� 5      -   5        $ ModeList��5      $y�	Collapsed=�6)   $	��100��6@   $��5,5,0,5q�6L   6v   5      �� 5   
   -   5        $��Show/Hide Settings��5      ���#CE353D
�������$��35��6V   $	��180��6b   
�������$	��5,0q�6�   $��0,5,5,0q�6�   $��Right=�6�   $��2��6�   � 5   
   $��1��6   $��3��6    5      �� 5      $	��50* �6#   �� 5      $��3* �6#   �� 5       $	��53* �6#   .+�5"      �� -   5         ��Name$ 	LeftImage��5"      $��0��6'   $��5,5,18,0q�67   $��2��6I   �� 5#      -   5        $ 
RightImage��5#      $��2��6(   $��5q�68    K� ZIndex$ 1��6C   �� 5%      -	   5        $��SettingsPanel��5%   �   $��2��6�   $ 2��6�   ���#F3F2EF$	��Top=�6�   $��5,5,5,0q�6�   y�6   ��    Settings �� Source# [   �� Path$% Default.SettingsPanelVisibility�	 �� Mode$	 TwoWay=�.+�5&      �� $��5q�6   $��0��6'   $
��Bold"�64   $��Settings��6F   � 5'       5(      �� 5)      $
��Auto �6+   �� 5*      $��* �6+    5,      �� 5-      $
��Auto �6(   �� 5.      $
��Auto �6(   �� 5/      $
��Auto �6(   �� 50      $
��Auto �6(   �� 51      $
��Auto �6(   �� 52      $
��Auto �6(   �� 53      $
��Auto �6(   �� 54      $
��Auto �6(   �� 55      $
��Auto �6(   .+�57      �� $��5q�6    $��0��6+   $��0��68   $��Face API Key:��6H   �� 58      $��5q�6    $��0��6+   $��1��68   ��6H   �� # [  $ Default.FaceAPIKey�$	 TwoWay=��� 59      $��5q�6    $��1��6+   $��0��68   $��Face API Host:��6H   �� 5:      $��5q�6    $��1��6+   $��1��68   ��6H   �� # [  $ Default.FaceAPIHost�$	 TwoWay=��� 5;      $y�	Collapsed=�6    $��5q�67   $��2��6B   $��0��6O   $��Emotion API Key:��6_   �� 5<      $y�	Collapsed=�6    $��5q�67   $��2��6B   $��1��6O   ��6_   �� # [  $ Default.EmotionAPIKey�$	 TwoWay=��� 5=      $��5q�6    $��3��6+   $y�	Collapsed=�68   $��0��6O   $��Emotion API Host:��6_   �� 5>      $��5q�6    $��3��6+   $y�	Collapsed=�68   $��1��6O   ��6_   �� # [  $ Default.EmotionAPIHost�$	 TwoWay=��� 5?      $��5q�6    $��4��6+   $y�	Collapsed=�68   $��0��6O   $��Vision API Key:��6_   �� 5@      $��5q�6    $��4��6+   $y�	Collapsed=�68   $��1��6O   ��6_   �� # [  $ Default.VisionAPIKey�$	 TwoWay=��� 5A      $��5q�6    $��5��6+   $y�	Collapsed=�68   $��0��6O   $��Vision API Host:��6_   �� 5B      $��5q�6    $��5��6+   $y�	Collapsed=�68   $��1��6O   ��6_   �� # [  $ Default.VisionAPIHost�$	 TwoWay=��� 5C      $y�	Collapsed=�6    $��5q�67   $��6��6B   $��2��6O   $
��Wrap=�6c   �� 6x   $�Get API keys at ��6|   �� 6�   -
   5        $(��"http://www.microsoft.com/cognitive?�5C   �   .�6�   �� $�microsoft.com/cognitive��6  6$  60  �� 5D      $��5q�6    $��7��6+   $��0��68   $��API Call Interval:��6H   �� 5E      $��5q�6    $��7��6+   $��1��68   ��6H   �� # [  $ Default.AnalysisInterval�$	 TwoWay=��� 5F      $y�	Collapsed=�6    $��5q�67   $��8��6B   $��2��6O   $
��Wrap=�6c   �� 6x   $
��Bold"�6|   $�Caution��6�   �� 6�   $W�Q: Each video frame submitted to an API will count towards your transaction limit.��6�   6�   5G      �� 5H      $y�	Collapsed=�6   $��5,0,5,5q�63   �� 5I      $�Face API calls made = ��6   �� 6:   �6>   �� # [  $ Default.FaceAPICallCount��� 6�   �� 5J      $�Emotion API calls made = ��6   �� 6=   �6A   �� # [  $! Default.EmotionAPICallCount��� 6�   �� 5K      $�Vision API calls made = ��6   �� 6<   �6@   �� # [  $  Default.VisionAPICallCount�5L      �� 5M      $��5q�6   
 �� ContentStringFormat$"
 Automatically stop after {0}��6{    d� 	IsChecked 6&   �� # [  $ Default.AutoStopEnabled���6�   �� # [  $ Default.AutoStopTime��� 5N      -   5        ���#CE353D$��30��5N   .   
�������$��5q�6M   $
��Save��6X   6g   5O      5P      �� 5Q   
   -   5         ��Name$ InformationText��5Q      
������� �� TextAlignment$ Center=�6@   $��Center=�6W   $��Center=�6r   $��50(�6�   $��DemiBold"�6�   $y�Visible=�6�   $@��:Project Auspex dfdfd dfdfjd fkjdkf dkf kdjf dkjf dkjfkjd f��6�   $��1��6
  $��1��6  $��10,205,5,217q�6'  $��2��6=  �� 5R   
   -   5        $��MessageArea��5R      $��5,10,5,9q�6'   $��2��69   $��Center=�6F   $��3��6a   5S      5T      