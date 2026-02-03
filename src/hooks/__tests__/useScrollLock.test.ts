import { describe, it, expect } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useScrollLock } from '../useScrollLock.ts';

describe('useScrollLock', () => {
  it('sets body.style.position to fixed when locked', () => {
    renderHook(() => useScrollLock(true));
    expect(document.body.style.position).toBe('fixed');
  });

  it('restores body styles when unlocked', () => {
    const { unmount } = renderHook(() => useScrollLock(true));
    expect(document.body.style.position).toBe('fixed');
    unmount();
    expect(document.body.style.position).toBe('');
  });

  it('does nothing when initially false', () => {
    renderHook(() => useScrollLock(false));
    expect(document.body.style.position).toBe('');
  });
});
