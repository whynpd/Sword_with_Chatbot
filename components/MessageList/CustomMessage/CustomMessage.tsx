import {
  ReactionSelector,
  ReactionsList,
  useMessageContext,
  useChannelStateContext
} from 'stream-chat-react';
import Image from 'next/image';
import { useState } from 'react';
import MessageOptions from './MessageOptions';
import { PencilIcon } from '@heroicons/react/24/outline';

export default function CustomMessage(): JSX.Element {
  const { message } = useMessageContext();
  const { channel } = useChannelStateContext();
  const [showOptions, setShowOptions] = useState(false);
  const [showReactions, setShowReactions] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editText, setEditText] = useState(message.text || '');

  const handleEdit = () => {
    setIsEditing(true);
    setEditText(message.text || '');
    setShowOptions(false);
  };

  const handleSaveEdit = async () => {
    if (editText.trim() === message.text) {
      setIsEditing(false);
      return;
    }

    try {
      if (!channel) return;

      await channel.update(
        { message: { id: message.id, text: editText } },
        { text: editText }
      );
      
      setIsEditing(false);
    } catch (error) {
      console.error('Error updating message:', error);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSaveEdit();
    }
    if (e.key === 'Escape') {
      setIsEditing(false);
      setEditText(message.text || '');
    }
  };

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

  const isMessageEdited = () => {
    if (!message.updated_at || !message.created_at) return false;
    const updatedAt = new Date(message.updated_at).getTime();
    const createdAt = new Date(message.created_at).getTime();
    return updatedAt > createdAt;
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
      <div className="flex-1">
        {showOptions && !isEditing && (
          <MessageOptions 
            showEmojiReactions={setShowReactions}
            onEdit={handleEdit}
          />
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
              {isMessageEdited() && ' (edited)'}
            </span>
          )}
        </div>
        {isEditing ? (
          <div className="flex items-end space-x-2">
            <textarea
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              onKeyDown={handleKeyPress}
              className="flex-1 p-2 text-sm text-gray-700 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 min-h-[60px] mt-1"
              autoFocus
            />
            <div className="flex space-x-2 mb-2">
              <button
                onClick={() => {
                  setIsEditing(false);
                  setEditText(message.text || '');
                }}
                className="px-2 py-1 text-sm text-gray-600 hover:text-gray-800"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveEdit}
                className="px-2 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                Save
              </button>
            </div>
          </div>
        ) : (
          <p className='text-sm text-gray-700'>{message.text}</p>
        )}
        {message.attachments?.map((attachment) => renderAttachment(attachment))}
        <ReactionsList />
      </div>
    </div>
  );

  function formatDate(date: Date | string): string {
    if (typeof date === 'string') {
      return new Date(date).toLocaleString('en-US', {
        dateStyle: 'medium',
        timeStyle: 'short',
      });
    }
    return date.toLocaleString('en-US', {
      dateStyle: 'medium',
      timeStyle: 'short',
    });
  }
}
