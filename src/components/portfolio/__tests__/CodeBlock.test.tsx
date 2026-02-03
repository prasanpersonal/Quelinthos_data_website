import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import CodeBlock from '../CodeBlock.tsx';
import type { CodeSnippet } from '../../../data/types.ts';

const snippets: CodeSnippet[] = [
  { language: 'sql', title: 'Create Table', description: 'Creates a table', code: 'CREATE TABLE foo;' },
  { language: 'python', title: 'Hello World', description: 'Prints hello', code: 'print("hello")' },
];

describe('CodeBlock', () => {
  it('renders tabs for multiple snippets', () => {
    render(<CodeBlock snippets={snippets} />);
    expect(screen.getByText('SQL')).toBeInTheDocument();
    expect(screen.getByText('Python')).toBeInTheDocument();
  });

  it('shows first snippet by default', () => {
    render(<CodeBlock snippets={snippets} />);
    expect(screen.getByText('Create Table')).toBeInTheDocument();
    expect(screen.getByText('CREATE TABLE foo;')).toBeInTheDocument();
  });

  it('switches content on tab click', () => {
    render(<CodeBlock snippets={snippets} />);
    fireEvent.click(screen.getByText('Python'));
    expect(screen.getByText('Hello World')).toBeInTheDocument();
    expect(screen.getByText('print("hello")')).toBeInTheDocument();
  });

  it('copy button is present', () => {
    render(<CodeBlock snippets={snippets} />);
    expect(screen.getByTitle('Copy to clipboard')).toBeInTheDocument();
  });

  it('returns null for empty snippets array', () => {
    const { container } = render(<CodeBlock snippets={[]} />);
    expect(container.innerHTML).toBe('');
  });
});
