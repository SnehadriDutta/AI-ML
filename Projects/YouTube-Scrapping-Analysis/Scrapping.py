from googleapiclient.discovery import build
import pandas as pd

def get_channel_data(youtube, channel_ids):
    list_playlist_ids = []
    request = youtube.channels().list(
        part='snippet,contentDetails,statistics',
        id=','.join(channel_ids)
    )
    response = request.execute()

    for i in range(len(response['items'])):
      playlist_id = response['items'][i]['contentDetails']['relatedPlaylists']['uploads']
      list_playlist_ids.append(playlist_id)

    return list_playlist_ids

def get_playlist_videoIds(youtube, playlist_id):
    video_ids = []

    request = youtube.playlistItems().list(
        part='contentDetails',
        playlistId=playlist_id,
        maxResults=50
    )

    response = request.execute()

    for i in range(len(response['items'])):
        video_ids.append(response['items'][i]['contentDetails']['videoId'])

    next_page_token = response.get('nextPageToken')
    more_pages = True

    while more_pages:
        if next_page_token is None:
            more_pages=False
        else:
            request = youtube.playlistItems().list(
                part='contentDetails',
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token
            )

            response = request.execute()

            for i in range(len(response['items'])):
                video_ids.append(response['items'][i]['contentDetails']['videoId'])

            next_page_token = response.get('nextPageToken')
    return video_ids

def get_video_details(youtube, video_ids):

    all_video_stats = []

    for i in range(0, len(video_ids), 50):

            request = youtube.videos().list(
                part='snippet,statistics',
                id= ','.join(video_ids[i:i+50])
            )
            response = request.execute()

            for video in response['items']:
                try:
                    video_stats = dict(Id= video['id'],
                                      Title = video['snippet']['title'],
                                      Published_Date = video['snippet']['publishedAt'],
                                      ThumbnailUrl= video['snippet']['thumbnails']['default']['url'],
                                      LikesCount= video['statistics']['likeCount'],
                                      ViewsCount= video['statistics']['viewCount'],
                                      CommentCount= video['statistics']['commentCount']
                                      )
                    all_video_stats.append(video_stats)
                except:
                      pass

    return all_video_stats

def create_dataset(api_key):

    channel_id = ['UCQ4FNww3XoNgqIlkBqEAVCg',
                  'UCMiJRAwDNSNzuYeN2uWa0pA',
                  'UCVYamHliCI9rw1tHR1xbkfw',
                  'UCXGgrKt94gR6lmN4aN3mYTg',
                  'UCvcRA2Hva1lULVf4GCouH8w',
                  'UCXuqSBlHAE6Xw-yeJA0Tunw'
                  ]

    youtube = build('youtube', 'v3', developerKey=api_key)
    playlist_ids = get_channel_data(youtube, channel_id)
    list_video_ids = []
    for id in playlist_ids:
        video_ids = get_playlist_videoIds(youtube, id)
        list_video_ids.extend(video_ids)
    video_details = get_video_details(youtube, list_video_ids)
    video_dataset = pd.DataFrame(video_details)

    return video_dataset

