import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import PainPointCard from '../PainPointCard.tsx';
import type { PainPoint } from '../../../data/types.ts';

const mockPainPoint: PainPoint = {
  id: 'test-pp',
  number: 1,
  title: 'Test Pain Point',
  subtitle: 'Test Subtitle',
  summary: 'A summary of the pain point',
  price: {
    present: { title: 'P', description: 'D', bullets: ['b'], severity: 'critical' },
    root: { title: 'R', description: 'D', bullets: ['b'], severity: 'high' },
    impact: { title: 'I', description: 'D', bullets: ['b'], severity: 'medium' },
    cost: { title: 'C', description: 'D', bullets: ['b'], severity: 'low' },
    expectedReturn: { title: 'E', description: 'D', bullets: ['b'], severity: 'high' },
  },
  implementation: {
    overview: 'Overview',
    prerequisites: ['Node.js'],
    steps: [{ stepNumber: 1, title: 'Step 1', description: 'Desc', codeSnippets: [] }],
    toolsUsed: ['Node.js'],
  },
  metrics: { annualCostRange: '$500K - $3M', roi: '6x', paybackPeriod: '3-4 months', investmentRange: '$80K - $150K' },
  tags: ['test'],
};

describe('PainPointCard', () => {
  it('renders title and summary', () => {
    render(<PainPointCard painPoint={mockPainPoint} index={0} accentColor="neon-blue" onDiveDeeper={() => {}} />);
    expect(screen.getByText('Test Pain Point')).toBeInTheDocument();
    expect(screen.getByText('A summary of the pain point')).toBeInTheDocument();
  });

  it('renders PRICE mini gauge letters', () => {
    render(<PainPointCard painPoint={mockPainPoint} index={0} accentColor="neon-blue" onDiveDeeper={() => {}} />);
    expect(screen.getByText('P')).toBeInTheDocument();
    expect(screen.getByText('R')).toBeInTheDocument();
    expect(screen.getByText('I')).toBeInTheDocument();
    expect(screen.getByText('C')).toBeInTheDocument();
    expect(screen.getByText('E')).toBeInTheDocument();
  });

  it('calls onDiveDeeper on button click', () => {
    const onDiveDeeper = vi.fn();
    render(<PainPointCard painPoint={mockPainPoint} index={0} accentColor="neon-blue" onDiveDeeper={onDiveDeeper} />);
    fireEvent.click(screen.getByRole('button'));
    expect(onDiveDeeper).toHaveBeenCalledTimes(1);
  });
});
