import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import Portfolio from '../Portfolio.tsx';
import { allCategories } from '../../data/categories/index.ts';

describe('Portfolio', () => {
  it('renders "THE INSIGHT SUITE" heading', () => {
    render(<Portfolio />);
    expect(screen.getByText('THE INSIGHT SUITE')).toBeInTheDocument();
  });

  it('displays correct pain point count text', () => {
    render(<Portfolio />);
    const total = allCategories.reduce((s, c) => s + c.painPoints.length, 0);
    expect(screen.getByText(new RegExp(`${total} Pain Points Identified`))).toBeInTheDocument();
  });

  it('renders all 10 category cards', () => {
    render(<Portfolio />);
    for (const cat of allCategories) {
      expect(screen.getByText(cat.title)).toBeInTheDocument();
    }
  });
});
