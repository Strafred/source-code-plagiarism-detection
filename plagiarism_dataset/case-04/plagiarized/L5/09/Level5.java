public class Level5 {
    public static void main(String[] args) {
        System.out.println("Miles\t\tKilometers");
        System.out.println("-------------------------------");
        convert();
    }
    private static void convert() {
        for(int mill = 1; mill <=10; mill++)
            System.out.println(mill + "\t\t" + mill * 1.609);
    }
}
