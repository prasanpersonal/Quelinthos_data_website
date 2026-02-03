import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import RoiCards from '../RoiCards.tsx';
import type { PainPointMetrics } from '../../../data/types.ts';

const mockMetrics: PainPointMetrics = {
  annualCostRange: '$500K - $3M',
  roi: '6x',
  paybackPeriod: '3-4 months',
  investmentRange: '$80K - $150K',
};

describe('RoiCards', () => {
  it('renders 3 metric cards', () => {
    render(<RoiCards metrics={mockMetrics} />);
    expect(screen.getByText('Annual Cost')).toBeInTheDocument();
    expect(screen.getByText('Investment')).toBeInTheDocument();
    expect(screen.getByText('ROI')).toBeInTheDocument();
  });

  it('displays correct metric values', () => {
    render(<RoiCards metrics={mockMetrics} />);
    expect(screen.getByText('$500K - $3M')).toBeInTheDocument();
    expect(screen.getByText('$80K - $150K')).toBeInTheDocument();
    expect(screen.getByText('6x return')).toBeInTheDocument();
  });
});
