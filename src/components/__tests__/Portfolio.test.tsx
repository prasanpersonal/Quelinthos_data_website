import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import Portfolio from '../Portfolio.tsx';
import { allCategories } from '../../data/categories/index.ts';

describe('Portfolio', () => {
  it('renders "PROBLEMS WE SOLVE" heading', () => {
    render(<Portfolio />);
    expect(screen.getByText('PROBLEMS WE SOLVE')).toBeInTheDocument();
  });

  it('displays correct pain point count text', () => {
    render(<Portfolio />);
    const total = allCategories.reduce((s, c) => s + c.painPoints.length, 0);
    expect(screen.getByText(new RegExp(`${total} specific data pain points`))).toBeInTheDocument();
  });

  it('renders all 10 category cards', () => {
    render(<Portfolio />);
    for (const cat of allCategories) {
      expect(screen.getByText(cat.title)).toBeInTheDocument();
    }
  });
});
