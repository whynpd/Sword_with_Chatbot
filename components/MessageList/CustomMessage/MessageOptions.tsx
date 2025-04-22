import { ArrowUturnLeft, Emoji, Thread } from '@/components/ChannelList/Icons';
import { PencilIcon } from '@heroicons/react/24/outline';
import { Dispatch, SetStateAction } from 'react';

export default function MessageOptions({
  showEmojiReactions,
  onEdit,
}: {
  showEmojiReactions: Dispatch<SetStateAction<boolean>>;
  onEdit: () => void;
}): JSX.Element {
  return (
    <div className='absolute flex items-center -top-4 right-2 rounded-md bg-gray-50 border-2 border-gray-200'>
      <button
        className='p-1 transition-colors duration-200 ease-in-out hover:bg-gray-200'
        onClick={() => showEmojiReactions((currentValue) => !currentValue)}
      >
        <Emoji className='w-6 h-6' />
      </button>
      <button
        className='p-1 transition-colors duration-200 ease-in-out hover:bg-gray-200'
        onClick={onEdit}
      >
        <PencilIcon className='w-6 h-6' />
      </button>
      <button className='p-1 transition-colors duration-200 ease-in-out hover:bg-gray-200'>
        <ArrowUturnLeft className='w-6 h-6' />
      </button>
      <button className='p-1 transition-colors duration-200 ease-in-out hover:bg-gray-200'>
        <Thread className='w-6 h-6' />
      </button>
    </div>
  );
}
