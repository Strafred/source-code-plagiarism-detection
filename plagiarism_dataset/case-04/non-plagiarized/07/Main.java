public class Main {
    public static void main(String[] args) {
        int miles = 1;
        System.out.print("Miles    Kilometer");
        for(int i=0;i<10;i++)
        {
            double kilo = miles*1.609;
            System.out.println("");
            System.out.print(miles+"        "+kilo);
            miles++;
        }
    }
}
