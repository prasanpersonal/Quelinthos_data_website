import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import IconResolver from '../IconResolver.tsx';

describe('IconResolver', () => {
  it('renders SVG for valid icon name', () => {
    const { container } = render(<IconResolver name="Home" />);
    const svg = container.querySelector('svg');
    expect(svg).toBeInTheDocument();
  });

  it('returns null for invalid icon name', () => {
    const { container } = render(<IconResolver name="NonExistentIcon" />);
    expect(container.innerHTML).toBe('');
  });

  it('passes size prop through', () => {
    const { container } = render(<IconResolver name="Home" size={32} />);
    const svg = container.querySelector('svg');
    expect(svg).toBeInTheDocument();
    expect(svg?.getAttribute('width')).toBe('32');
    expect(svg?.getAttribute('height')).toBe('32');
  });

  it('passes className prop through', () => {
    const { container } = render(<IconResolver name="Home" className="text-red-500" />);
    const svg = container.querySelector('svg');
    expect(svg).toHaveClass('text-red-500');
  });
});
