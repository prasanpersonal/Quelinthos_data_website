import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import Home from '../Home.tsx';

describe('Home', () => {
  it('renders hero badge "Data Consulting That Delivers"', () => {
    render(<Home />);
    expect(screen.getByText('Data Consulting That Delivers')).toBeInTheDocument();
  });

  it('renders heading "QUELINTHOS" and "DYNAMICS"', () => {
    render(<Home />);
    expect(screen.getByText('QUELINTHOS')).toBeInTheDocument();
    expect(screen.getByText('DYNAMICS')).toBeInTheDocument();
  });

  it('renders feature pills from data', () => {
    render(<Home />);
    expect(screen.getByText('Data Engineering')).toBeInTheDocument();
    expect(screen.getByText('Analytics & BI')).toBeInTheDocument();
    expect(screen.getByText('AI & ML Solutions')).toBeInTheDocument();
    expect(screen.getByText('Dashboards & Reporting')).toBeInTheDocument();
  });
});
