import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import CategoryCard from '../CategoryCard.tsx';
import type { Category } from '../../../data/types.ts';

const mockCategory: Category = {
  id: 'test-cat',
  number: 1,
  title: 'Test Category',
  shortTitle: 'Test',
  description: 'A test category description',
  icon: 'Home',
  accentColor: 'neon-blue',
  painPoints: [
    {
      id: 'test-pp',
      number: 1,
      title: 'Test Pain Point',
      subtitle: 'Subtitle',
      summary: 'Summary',
      price: {
        present: { title: 'P', description: 'D', bullets: ['b'], severity: 'high' },
        root: { title: 'R', description: 'D', bullets: ['b'], severity: 'high' },
        impact: { title: 'I', description: 'D', bullets: ['b'], severity: 'high' },
        cost: { title: 'C', description: 'D', bullets: ['b'], severity: 'high' },
        expectedReturn: { title: 'E', description: 'D', bullets: ['b'], severity: 'medium' },
      },
      implementation: {
        overview: 'Overview',
        prerequisites: ['Node.js'],
        steps: [{ stepNumber: 1, title: 'Step 1', description: 'Desc', codeSnippets: [] }],
        toolsUsed: ['Node.js'],
      },
      metrics: { annualCostRange: '$500K - $3M', roi: '6x', paybackPeriod: '3-4 months', investmentRange: '$80K - $150K' },
      tags: ['test'],
    },
  ],
};

describe('CategoryCard', () => {
  it('renders category title and description', () => {
    render(<CategoryCard category={mockCategory} index={0} onClick={() => {}} />);
    expect(screen.getByText('Test Category')).toBeInTheDocument();
    expect(screen.getByText('A test category description')).toBeInTheDocument();
  });

  it('renders pain point count badge', () => {
    render(<CategoryCard category={mockCategory} index={0} onClick={() => {}} />);
    expect(screen.getByText('1 pain point identified')).toBeInTheDocument();
  });

  it('calls onClick when clicked', () => {
    const onClick = vi.fn();
    render(<CategoryCard category={mockCategory} index={0} onClick={onClick} />);
    // Click on the card container
    fireEvent.click(screen.getByText('Test Category'));
    expect(onClick).toHaveBeenCalled();
  });
});
