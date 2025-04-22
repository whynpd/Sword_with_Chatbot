import {
  ReactionSelector,
  ReactionsList,
  useMessageContext,
} from 'stream-chat-react';
import Image from 'next/image';
import { useState } from 'react';
import MessageOptions from './MessageOptions';

export default function CustomMessage(): JSX.Element {
  const { message } = useMessageContext();
  const [showOptions, setShowOptions] = useState(false);
  const [showReactions, setShowReactions] = useState(false);

  const renderAttachment = (attachment: any) => {
    if (attachment.type === 'file') {
      return (
        <div key={attachment.asset_url} className="mt-2">
          <a 
            href={attachment.asset_url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center space-x-2 p-2 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors duration-200 max-w-fit"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <div className="flex flex-col">
              <span className="text-sm font-medium text-blue-600">{attachment.title}</span>
              <span className="text-xs text-gray-500">
                {formatFileSize(attachment.file_size)}
              </span>
            </div>
          </a>
        </div>
      );
    }
    return null;
  };

  const formatFileSize = (bytes: number) => {
    if (!bytes) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
  };

  return (
    <div
      onMouseEnter={() => setShowOptions(true)}
      onMouseLeave={() => setShowOptions(false)}
      className='flex relative space-x-2 p-2 rounded-md transition-colors ease-in-out duration-200 hover:bg-gray-100'
    >
      <Image
        className='rounded-full aspect-square object-cover w-10 h-10'
        width={40}
        height={40}
        src={message.user?.image || 'https://getstream.io/random_png/'}
        alt='User avatar'
      />
      <div>
        {showOptions && (
          <MessageOptions showEmojiReactions={setShowReactions} />
        )}
        {showReactions && (
          <div className='absolute'>
            <ReactionSelector />
          </div>
        )}
        <div className='space-x-2'>
          <span className='font-semibold text-sm text-black'>
            {message.user?.name}
          </span>
          {message.updated_at && (
            <span className='text-xs text-gray-600'>
              {formatDate(message.updated_at)}
            </span>
          )}
        </div>
        <p className='text-sm text-gray-700'>{message.text}</p>
        {message.attachments?.map((attachment) => renderAttachment(attachment))}
        <ReactionsList />
      </div>
    </div>
  );

  function formatDate(date: Date | string): string {
    if (typeof date === 'string') {
      return date;
    }
    return `${date.toLocaleString('en-US', {
      dateStyle: 'medium',
      timeStyle: 'short',
    })}`;
  }
}
