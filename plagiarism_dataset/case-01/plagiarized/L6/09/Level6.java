public class Level6 
{
    public static void main(String[] args) 
    {
        String input = "Welcome to Java\n";
        keluaran(input);
    }
    private static void keluaran(String kal) 
    {
        int limit=0; 
        while(limit!=5) 
        {
            System.out.print(kal);
            limit++;
        }
    }
}
