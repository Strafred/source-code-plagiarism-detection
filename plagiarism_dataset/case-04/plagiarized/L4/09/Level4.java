public class Level4 
{
    public static void main(String[] args) 
    {
        System.out.println("Miles\t\tKilometers");
        System.out.println("-------------------------------");
        tabelKonversi();
    }
    private static void tabelKonversi() 
    {
        int mil = 1;
        while (mil <= 10) 
        {
            System.out.println(mil + "\t\t" + mil * 1.609);
            mil++;
        }
    }
}
